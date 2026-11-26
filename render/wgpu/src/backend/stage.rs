//! Stage packets, reliable cache writes and submission to the reserved output.
use super::*;
use crate::buffer_pool::{AlwaysCompatible, PoolEntry};
use crate::render_snapshot::{FramePacket, PacketDraw, PreparedSnapshot, SnapshotBindings};

type SavedStage = PoolEntry<
    (
        crate::gpu_resources::Texture,
        crate::gpu_resources::TextureView,
    ),
    AlwaysCompatible,
>;

pub(super) struct PendingStage {
    key: PacketKey,
    draw: std::sync::Weak<Option<PacketDraw>>,
    content: StageContent,
}

enum StageContent {
    Live(SnapshotBindings),
    Encoded(SavedStage),
    Failed(RenderFailure),
}

fn stage_label() -> String {
    ruffle_render::profile::next_frame()
        .map(|n| format!("Stage {n}"))
        .unwrap_or_else(|| "Frame commands".into())
}

impl PendingStage {
    pub(super) fn cancelled(&self) -> bool {
        self.draw.strong_count() == 0
    }
}

impl<T: RenderTarget> WgpuRenderBackend<T> {
    pub(super) fn stage_draw_route(
        &self,
        commands: &CommandList,
        format: wgpu::TextureFormat,
    ) -> StageDrawRoute {
        let device = self.descriptors.runtime.generation();
        if matches!(
            format,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
        ) && crate::graph_encoder::SurfaceCommands::pixel_aligned_caches(commands, device)
        {
            StageDrawRoute::AlignedBitmaps
        } else if PreparedSnapshot::direct_resident_draws(commands, device) {
            StageDrawRoute::Resident
        } else {
            StageDrawRoute::Graph
        }
    }

    pub(super) fn next_stage_generation(&self) -> Result<u64, RenderFailure> {
        self.packet_key
            .target_generation
            .checked_add(1)
            .ok_or_else(|| {
                RenderFailure::new(
                    RenderFailureKind::Internal,
                    "stage target generation exhausted",
                )
            })
    }

    pub(super) fn invalidate_stage_packets(&mut self, generation: u64) {
        self.pending_stage = None;
        self.last_stage_attachment = None;
        self.packet_key.target_generation = generation;
    }

    /// Consume the reliable cache writes while their VM inputs are live. The
    /// resulting picture can replace an unstarted picture on this endpoint.
    pub fn capture_stage_packet(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) -> Result<FramePacket, RenderFailure> {
        self.capture_stage(clear, commands, cache_entries, false)?
            .ok_or_else(|| {
                RenderFailure::new(
                    RenderFailureKind::BudgetExceeded,
                    "owned GPU payload queue is full",
                )
            })
    }

    /// Realtime callers can retry after transient pressure. If the snapshot would
    /// never fit, encode it now and keep only the native image until dispatch.
    pub fn try_capture_stage_packet(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) -> Result<Option<FramePacket>, RenderFailure> {
        self.capture_stage(clear, commands, cache_entries, true)
    }

    fn capture_stage(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
        allow_oversized_draw: bool,
    ) -> Result<Option<FramePacket>, RenderFailure> {
        use ruffle_render::profile::{NativePhase, native_time};

        self.descriptors.runtime.check_health()?;
        self.pending_stage = None;
        // The VM has already marked these cache entries current. Their writes
        // stay on the ordered encoder even if the final picture has to wait.
        {
            let _timer = native_time(NativePhase::StageCacheEncode);
            self.encode_stage_caches(cache_entries)?;
        }
        let route = self.stage_draw_route(&commands, self.target.format());
        let prepared = {
            let _timer = native_time(NativePhase::StageSnapshotCapture);
            PreparedSnapshot::capture_for_device(&commands, Some(self.packet_key.device))?
        };
        let bytes = prepared.packet_bytes();
        let oversized =
            allow_oversized_draw && bytes > self.descriptors.runtime.config().staging_limit;
        let charge = if oversized { 0 } else { bytes };
        let reservation = self
            .descriptors
            .runtime
            .reserve_payload(crate::gpu_runtime::PayloadKind::Frame, charge);
        let payload = match reservation {
            Ok(ticket) => ticket,
            Err(error) if error.kind() == RenderFailureKind::BudgetExceeded => return Ok(None),
            Err(error) => return Err(error),
        };
        let draw = PacketDraw {
            clear,
            snapshot: prepared.snapshot,
            route,
        };
        let bindings = prepared.bindings;
        // Context3D submits through an independent encoder. Order the old
        // pixels on the shared queue before that encoder can reuse its target.
        let stage3d = bindings.reads_stage3d;
        let packet = if oversized || stage3d {
            let saved = self.encode_saved_stage(draw, bindings, stage3d)?;
            self.stage_packet(None, StageContent::Encoded(saved), Some(payload))?
        } else {
            self.stage_packet(Some(draw), StageContent::Live(bindings), Some(payload))?
        };
        Ok(Some(packet))
    }

    fn stage_packet(
        &mut self,
        draw: Option<PacketDraw>,
        content: StageContent,
        payload: Option<crate::gpu_runtime::OwnedPayload>,
    ) -> Result<FramePacket, RenderFailure> {
        self.packet_key.packet = self.packet_key.packet.checked_add(1).ok_or_else(|| {
            RenderFailure::new(
                RenderFailureKind::Internal,
                "stage packet sequence exhausted",
            )
        })?;
        self.packet_key.reservation = self.target.frame_reservation();
        self.packet_key.operation = self.active_frame.operations.applied();
        let draw = Arc::new(draw);
        self.pending_stage = Some(PendingStage {
            key: self.packet_key,
            draw: Arc::downgrade(&draw),
            content,
        });
        Ok(FramePacket {
            payload,
            key: self.packet_key,
            draw,
        })
    }

    /// Submit a captured picture into the currently reserved stage output.
    /// Rejection of an obsolete packet leaves the successor's reservation live.
    pub fn submit_stage_packet(
        &mut self,
        packet: FramePacket,
    ) -> Result<ruffle_render::backend::FrameSubmission, RenderFailure> {
        let frame = self.packet_output(&packet)?;
        self.submit_packet_to(frame, packet)
    }

    /// Resolve owner-local bindings here, then hand the executor the recorded
    /// native commands with their queue position. The target frame stays local.
    pub fn encode_stage_packet(
        &mut self,
        packet: FramePacket,
    ) -> Result<(crate::gpu_resources::StageWork, T::Frame), RenderFailure> {
        let frame = self.packet_output(&packet)?;
        self.draw_packet(&frame, packet)?;
        Ok((self.take_stage_encoder()?.into_stage_work()?, frame))
    }

    fn packet_output(&mut self, packet: &FramePacket) -> Result<T::Frame, RenderFailure> {
        self.descriptors.runtime.check_health()?;
        if self.pending_stage.as_ref().map(|p| p.key) != Some(packet.key)
            || self.target.frame_reservation() != packet.key.reservation
        {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "obsolete stage packet",
            ));
        }
        self.take_stage_output()
    }

    fn submit_packet_to(
        &mut self,
        frame: T::Frame,
        packet: FramePacket,
    ) -> Result<ruffle_render::backend::FrameSubmission, RenderFailure> {
        self.draw_packet(&frame, packet)?;
        self.finish_stage(frame)
    }

    fn draw_packet(&mut self, frame: &T::Frame, packet: FramePacket) -> Result<(), RenderFailure> {
        let mut payload = packet.payload;
        if let Some(ticket) = &mut payload {
            ticket.begin_encoding();
        }
        let pending = self.pending_stage.take().expect("matched pending stage");
        match pending.content {
            StageContent::Failed(error) => return Err(error),
            StageContent::Live(bindings) => {
                let draw = Arc::unwrap_or_clone(packet.draw).expect("live stage commands");
                let clear = draw.clear;
                let commands = self.plan_stage_packet(draw, bindings)?;
                self.encode_stage_draw(frame.view(), clear, commands, stage_label())?;
            }
            StageContent::Encoded(saved) => {
                let _pass = self
                    .descriptors
                    .device
                    .pass_context(crate::gpu_resources::diagnostics::Context::Stage);
                run_copy_pipeline(
                    &self.descriptors,
                    self.target.format(),
                    frame.view(),
                    &saved.1,
                    1,
                    &mut self
                        .profiler
                        .scope(stage_label(), self.active_frame.encoder()),
                )?;
            }
        }
        Ok(())
    }

    fn plan_stage_packet(
        &mut self,
        draw: PacketDraw,
        bindings: SnapshotBindings,
    ) -> Result<crate::graph_encoder::SurfaceCommands, RenderFailure> {
        use crate::graph_encoder::SurfaceCommands;
        let _prepare = ruffle_render::profile::native_time(
            ruffle_render::profile::NativePhase::CommandPreparation,
        );
        let mut prepared = PreparedSnapshot {
            snapshot: draw.snapshot,
            bindings,
        };
        match draw.route {
            StageDrawRoute::AlignedBitmaps | StageDrawRoute::Resident => {
                prepared.validate(self)?;
                let commands = prepared.direct_commands()?;
                Ok(match draw.route {
                    StageDrawRoute::AlignedBitmaps => SurfaceCommands::CachedBitmaps(commands),
                    _ => SurfaceCommands::ResidentDraws(commands),
                })
            }
            StageDrawRoute::Graph => {
                if let Some(scale) = self.stage_projection() {
                    prepared
                        .snapshot
                        .project_stage(scale, self.descriptors.backend);
                }
                let graph = crate::graph_encoder::GraphEncoder::from_prepared(
                    self,
                    prepared,
                    ruffle_render::region::RenderRect {
                        x: 0,
                        y: 0,
                        width: self.target.width(),
                        height: self.target.height(),
                    },
                    self.surface.sample_count(),
                    crate::pass_graph::Load::Clear(draw.clear),
                )?;
                Ok(SurfaceCommands::Graph(graph))
            }
        }
    }

    pub(super) fn preserve_stage_before_write(
        &mut self,
        destination: &BitmapHandle,
    ) -> Result<(), RenderFailure> {
        let Some(pending) = &self.pending_stage else {
            return Ok(());
        };
        let Some(draw) = pending.draw.upgrade() else {
            self.pending_stage = None;
            return Ok(());
        };
        let StageContent::Live(bindings) = &pending.content else {
            return Ok(());
        };
        if !bindings.reads(crate::render_snapshot::bitmap_version(destination)?) {
            return Ok(());
        }

        let mut pending = self.pending_stage.take().expect("live stage");
        let StageContent::Live(bindings) = pending.content else {
            unreachable!()
        };
        let draw = Arc::unwrap_or_clone(draw).expect("live stage commands");
        let result = self.encode_saved_stage(draw, bindings, false);
        let status = result.as_ref().map(|_| ()).map_err(Clone::clone);
        pending.content = match result {
            Ok(saved) => StageContent::Encoded(saved),
            Err(error) => StageContent::Failed(error),
        };
        self.pending_stage = Some(pending);
        status
    }

    fn encode_saved_stage(
        &mut self,
        draw: PacketDraw,
        bindings: SnapshotBindings,
        submit_before_return: bool,
    ) -> Result<SavedStage, RenderFailure> {
        // Encode V before a write to V+1, including bitmap fills already
        // baked into resident shape bind groups, without a CPU readback.
        let clear = draw.clear;
        let commands = self.plan_stage_packet(draw, bindings)?;
        let saved = self.offscreen_texture_pool.get_texture(
            &self.descriptors,
            wgpu::Extent3d {
                width: self.target.width(),
                height: self.target.height(),
                depth_or_array_layers: 1,
            },
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            self.target.format(),
            1,
        )?;
        self.encode_stage_draw(&saved.1, clear, commands, "Preserve stage pixels".into())?;
        if submit_before_return {
            self.active_frame.submit_direct(&self.descriptors)?;
        }
        Ok(saved)
    }

    fn take_stage_output(&mut self) -> Result<T::Frame, RenderFailure> {
        self.stage_frame.take().ok_or_else(|| {
            RenderFailure::new(
                RenderFailureKind::Internal,
                "stage submission without a reserved output",
            )
        })
    }

    fn encode_stage_caches(
        &mut self,
        cache_entries: Vec<BitmapCacheEntry>,
    ) -> Result<(), RenderFailure> {
        self.descriptors.runtime.check_health()?;
        if let Some(tiles) = &mut self.filtered_tiles {
            tiles.begin_frame();
        }
        if let Some(inputs) = &self.descriptors.blend_inputs {
            inputs.begin_frame();
        }
        self.surface.display = self
            .display_enhancer
            .as_ref()
            .filter(|e| e.enabled())
            .cloned();
        self.encode_bitmap_caches(cache_entries, None)
    }

    pub(super) fn encode_bitmap_caches(
        &mut self,
        cache_entries: Vec<BitmapCacheEntry>,
        offscreen_quality: Option<StageQuality>,
    ) -> Result<(), RenderFailure> {
        let _pass = self
            .descriptors
            .device
            .pass_context(crate::gpu_resources::diagnostics::Context::Offscreen);
        let quality = offscreen_quality.unwrap_or(self.surface.quality());
        let display = self
            .surface
            .display
            .as_ref()
            .filter(|_| offscreen_quality.is_none())
            .cloned();
        for entry in cache_entries {
            let prepared = crate::render_snapshot::PreparedSnapshot::capture(&entry.commands)?;
            // Only stage caches participate in redraw history and filtered tile reuse.
            let cache = (&*entry.handle.0 as &dyn Any)
                .downcast_ref::<crate::bitmap_cache::CacheImage>()
                .filter(|_| offscreen_quality.is_none());
            let texture = as_texture(&entry.handle, &self.descriptors.device)?;
            let valid = ruffle_render::region::RenderRect {
                x: 0,
                y: 0,
                width: texture.texture.width(),
                height: texture.texture.height(),
            };
            let (dirty, history) = if let Some(cache) = cache {
                cache.prepare_redraw(
                    &prepared.snapshot,
                    entry.clear,
                    quality,
                    &entry.filters,
                    self.descriptors.runtime.scheduler(),
                )
            } else {
                (ruffle_render::region::RenderRegion::Full, None)
            };
            let Some(dirty) = dirty.clip(valid) else {
                if let Some(cache) = cache {
                    cache.finish_redraw(history);
                }
                continue;
            };
            self.preserve_read_before_write(&entry.handle)?;
            let tile = if display.is_none() && self.filtered_tiles.is_some() && cache.is_some() {
                crate::filtered_tiles::Input::capture(
                    &prepared,
                    &entry.filters,
                    entry.clear,
                    quality,
                    self.surface.sample_count(),
                    valid.width,
                    valid.height,
                )
            } else {
                None
            };
            if let (Some(tiles), Some(input)) = (&mut self.filtered_tiles, &tile) {
                if let Some(id) = tiles.replay(
                    input,
                    &texture.texture,
                    self.active_frame.command_encoder.record(),
                ) {
                    ruffle_render::profile::work(|w| w.filtered_tile_hits += 1);
                    self.finish_bitmap_write(&entry.handle)?;
                    if let Some(cache) = cache {
                        cache
                            .tile_id
                            .store(id, std::sync::atomic::Ordering::Relaxed);
                        cache.finish_redraw(history);
                    }
                    self.active_frame.maybe_flush(&self.descriptors)?;
                    continue;
                }
                ruffle_render::profile::work(|w| w.filtered_tile_misses += 1);
            }
            let (input, output) = crate::bitmap_cache::filter_regions(dirty, valid, &entry.filters);
            let partial = input != valid;
            let surface = Surface::new(
                &self.descriptors,
                quality,
                input.width,
                input.height,
                texture.texture.format(),
            )?
            .with_origin([input.x, input.y])
            .with_display(display.clone());
            let mut graph = crate::graph_encoder::GraphEncoder::from_prepared(
                self,
                prepared,
                input,
                surface.sample_count(),
                crate::pass_graph::Load::Clear(entry.clear),
            )?;
            let filter_passes = if entry.filters.is_empty() {
                None
            } else {
                let max_groups = if self.descriptors.compute_filters {
                    self.descriptors.capabilities.max_compute_groups
                } else {
                    0
                };
                Some(graph.graph.take_filter_chain(
                    &entry.filters,
                    input,
                    texture.texture.format(),
                    max_groups,
                )?)
            };
            let commands = graph.into_commands()?;
            ruffle_render::profile::work(|w| {
                w.offscreen_targets += 1;
                w.offscreen_pixels += u64::from(input.width) * u64::from(input.height);
            });
            let color = wgpu::Color {
                r: f64::from(entry.clear.r) / 255.0,
                g: f64::from(entry.clear.g) / 255.0,
                b: f64::from(entry.clear.b) / 255.0,
                a: f64::from(entry.clear.a) / 255.0,
            };
            let destination = (!partial
                && self.descriptors.filter_cache_writeback
                    == crate::descriptors::FilterCacheWriteback::FilterOutput
                && texture.texture.format() == wgpu::TextureFormat::Rgba8Unorm
                && texture.texture.sample_count() == 1
                && texture
                    .texture
                    .usage()
                    .contains(wgpu::TextureUsages::RENDER_ATTACHMENT))
            .then(|| FilterSource::for_entire_texture(&texture.texture, &texture.view));
            // A standalone glow or shadow still samples the unfiltered image in its final pass.
            let separate_source = destination.is_some()
                && matches!(
                    entry.filters.as_slice(),
                    [Filter::GlowFilter(_) | Filter::DropShadowFilter(_)]
                );
            let mode = if partial || separate_source {
                RenderTargetMode::FreshWithColor(color)
            } else {
                RenderTargetMode::ExistingWithView(texture.view.clone(), color)
            };
            let mut scope = self
                .profiler
                .scope("Draw to CAB", self.active_frame.command_encoder.record());
            let mut target = surface.draw_commands(
                mode,
                &self.descriptors,
                &self.meshes,
                commands,
                &mut self.active_frame.staging_belt,
                &mut self.chunk_buffers,
                &mut scope,
                LayerRef::None,
                &mut self.offscreen_texture_pool,
            )?;
            target.ensure_cleared(&mut scope);
            let mut filtered = false;
            if let Some(plan) = filter_passes
                && let Some(result) = self.descriptors.filters.apply_chain(
                    &self.descriptors,
                    &mut scope.scope("Filter chain"),
                    &mut self.offscreen_texture_pool,
                    &mut self.active_frame.staging_belt,
                    FilterSource::for_entire_texture(target.color_texture(), target.color_view()),
                    &entry.filters,
                    plan,
                    destination,
                )?
            {
                target = result;
                filtered = true;
            }
            if partial {
                scope.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: target.color_texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: output.x - input.x,
                            y: output.y - input.y,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: output.x,
                            y: output.y,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: output.width,
                        height: output.height,
                        depth_or_array_layers: 1,
                    },
                );
            } else if filtered && target.color_texture() == &texture.texture {
                ruffle_render::profile::work(|w| w.filter_cache_direct_outputs += 1);
            } else if filtered {
                let source = target.color_texture();
                let destination = &texture.texture;
                let mut copy = scope.scope("Copy filtered to CAB");
                if self.descriptors.filter_cache_writeback
                    == crate::descriptors::FilterCacheWriteback::Transfer
                    && source.format() == wgpu::TextureFormat::Rgba8Unorm
                    && source.format() == destination.format()
                    && source.size() == destination.size()
                    && source.sample_count() == 1
                    && destination.sample_count() == 1
                    && source.usage().contains(wgpu::TextureUsages::COPY_SRC)
                    && destination.usage().contains(wgpu::TextureUsages::COPY_DST)
                {
                    // Drawing into an existing cache can overwrite its entire source
                    // image. Restore the full filtered result even when its halo
                    // expanded a smaller dirty region to cover the whole input.
                    copy.copy_texture_to_texture(
                        source.as_image_copy(),
                        destination.as_image_copy(),
                        source.size(),
                    );
                    ruffle_render::profile::work(|w| {
                        w.filter_cache_transfers += 1;
                        w.copy_bytes += u64::from(valid.width) * u64::from(valid.height) * 4;
                    });
                } else {
                    ruffle_render::profile::work(|w| w.filter_cache_draw_copies += 1);
                    run_copy_pipeline(
                        &self.descriptors,
                        destination.format(),
                        &texture.view,
                        target.color_view(),
                        source.sample_count(),
                        &mut copy,
                    )?;
                }
            }
            drop(scope);
            self.finish_bitmap_write(&entry.handle)?;
            if let (Some(tiles), Some(input), Some(cache)) = (&mut self.filtered_tiles, tile, cache)
                && let Some(id) = tiles.store(
                    input,
                    &texture.texture,
                    &self.descriptors.device,
                    self.active_frame.command_encoder.record(),
                )?
            {
                cache
                    .tile_id
                    .store(id, std::sync::atomic::Ordering::Relaxed);
            }
            if let Some(cache) = cache {
                cache.finish_redraw(history);
            }
            self.active_frame.maybe_flush(&self.descriptors)?;
        }

        Ok(())
    }

    fn stage_projection(&self) -> Option<ruffle_render::matrix::Matrix> {
        let size = self.surface.size();
        ((size.width, size.height) != (self.viewport.width, self.viewport.height)).then(|| {
            ruffle_render::matrix::Matrix::scale(
                size.width as f32 / self.viewport.width as f32,
                size.height as f32 / self.viewport.height as f32,
            )
        })
    }

    fn encode_stage_draw(
        &mut self,
        view: &crate::gpu_resources::TextureView,
        clear: Color,
        graph: crate::graph_encoder::SurfaceCommands,
        label: String,
    ) -> Result<(), RenderFailure> {
        use crate::graph_encoder::SurfaceCommands;
        let graph = match graph {
            SurfaceCommands::ResidentDraws(list) | SurfaceCommands::CachedBitmaps(list)
                if self.stage_projection().is_some() =>
            {
                SurfaceCommands::ViewportDraws(list, self.viewport)
            }
            other => other,
        };
        let _pass = self
            .descriptors
            .device
            .pass_context(crate::gpu_resources::diagnostics::Context::Stage);
        let attachment = self.surface.draw_graph_and_copy_to(
            view,
            RenderTargetMode::FreshWithColor(wgpu::Color {
                r: f64::from(clear.r) / 255.0,
                g: f64::from(clear.g) / 255.0,
                b: f64::from(clear.b) / 255.0,
                a: f64::from(clear.a) / 255.0,
            }),
            &self.descriptors,
            &mut self.active_frame.staging_belt,
            &mut self.chunk_buffers,
            &mut self
                .profiler
                .scope(label, self.active_frame.command_encoder.record()),
            &self.meshes,
            graph,
            LayerRef::None,
            &mut self.texture_pool,
        )?;
        self.last_stage_attachment = Some(attachment);
        Ok(())
    }

    fn finish_stage(
        &mut self,
        frame_output: T::Frame,
    ) -> Result<ruffle_render::backend::FrameSubmission, RenderFailure> {
        let encoder = self.take_stage_encoder()?;
        let index = {
            let _submit = ruffle_render::profile::time(ruffle_render::profile::Phase::RenderSubmit);
            self.target.submit_encoder(
                &self.descriptors.device,
                &self.descriptors.queue,
                self.descriptors.runtime.health(),
                encoder,
                frame_output,
            )?
        };
        self.confirm_stage_submission(index)?;
        Ok(ruffle_render::backend::FrameSubmission::Submitted)
    }

    fn take_stage_encoder(
        &mut self,
    ) -> Result<crate::gpu_resources::CommandEncoder, RenderFailure> {
        if let Some(e) = &self.surface.display {
            e.finish_frame(self.active_frame.command_encoder.record())?;
        }
        self.profiler
            .resolve_queries(self.active_frame.command_encoder.record());
        self.profiler
            .end_frame(&self.active_frame.command_encoder.raw)
            .map_err(|error| self.descriptors.runtime.health().record(error.into()))?;
        let (idle, peak, combined) = self.budget.stats();
        ruffle_render::profile::pool(idle, peak, combined);
        self.texture_pool.end_frame();
        self.offscreen_texture_pool.end_frame();
        self.active_frame.take_encoder(&self.descriptors)
    }

    pub fn confirm_stage_submission(
        &mut self,
        index: wgpu::SubmissionIndex,
    ) -> Result<(), RenderFailure> {
        self.last_submission = Some(index);
        self.active_frame.staging_belt.recall();
        ruffle_render::profile::submitted();
        self.collect_gpu_samples();

        self.descriptors.runtime.check_health()
    }

    pub(super) fn submit_inline_frame(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) -> Result<ruffle_render::backend::FrameSubmission, crate::gpu_runtime::RenderFailure> {
        let _render_backend =
            ruffle_render::profile::time(ruffle_render::profile::Phase::RenderBackend);
        let frame_output = self.take_stage_output()?;
        self.pending_stage = None;
        self.encode_stage_caches(cache_entries)?;
        let graph = if self.owned_frames || self.stage_projection().is_some() {
            // An immediately consumed, resident-only list may borrow its VM
            // pages until this call returns. Queued packets always own theirs.
            let route = self.stage_draw_route(&commands, self.target.format());
            match route {
                StageDrawRoute::Resident => {
                    crate::graph_encoder::SurfaceCommands::ResidentDraws(commands)
                }
                StageDrawRoute::AlignedBitmaps => {
                    crate::graph_encoder::SurfaceCommands::CachedBitmaps(commands)
                }
                StageDrawRoute::Graph => {
                    let prepared = PreparedSnapshot::capture_for_device(
                        &commands,
                        Some(self.packet_key.device),
                    )?;
                    self.plan_stage_packet(
                        PacketDraw {
                            clear,
                            route,
                            snapshot: prepared.snapshot,
                        },
                        prepared.bindings,
                    )?
                }
            }
        } else {
            self.prepare_commands(
                commands,
                self.target.width(),
                self.target.height(),
                self.surface.quality(),
                self.target.format(),
                crate::pass_graph::Load::Clear(clear),
            )?
        };
        self.encode_stage_draw(frame_output.view(), clear, graph, stage_label())?;
        self.finish_stage(frame_output)
    }
}
