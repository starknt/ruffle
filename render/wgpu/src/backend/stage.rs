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
    draw: std::sync::Weak<PacketDraw>,
    content: StageContent,
}

enum StageContent {
    Live(SnapshotBindings),
    Encoded(SavedStage),
    Failed(RenderFailure),
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

    pub(super) fn invalidate_stage_packets(&mut self) -> Result<(), RenderFailure> {
        self.pending_stage = None;
        self.packet_key.target_generation = self
            .packet_key
            .target_generation
            .checked_add(1)
            .ok_or_else(|| {
                RenderFailure::new(
                    RenderFailureKind::Internal,
                    "stage target generation exhausted",
                )
            })?;
        Ok(())
    }

    /// Consume the reliable cache writes while their VM inputs are live. The
    /// resulting picture can replace an unstarted picture on this endpoint.
    pub fn capture_stage_packet(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) -> Result<FramePacket, RenderFailure> {
        self.descriptors.runtime.check_health()?;
        self.pending_stage = None;
        self.encode_stage_caches(cache_entries)?;
        let route = self.stage_draw_route(&commands, self.target.format());
        let packet = self.capture_stage_commands(clear, &commands, route, true)?;
        if self.pending_stage.as_ref().is_some_and(
            |p| matches!(&p.content, StageContent::Live(bindings) if bindings.reads_stage3d),
        ) {
            // Context3D submits through an independent encoder. Order the old
            // pixels on the shared queue before that encoder can reuse its target.
            self.save_pending_stage(packet.draw.clone(), true)?;
        }
        Ok(packet)
    }

    fn capture_stage_commands(
        &mut self,
        clear: Color,
        commands: &CommandList,
        route: StageDrawRoute,
        queued: bool,
    ) -> Result<FramePacket, RenderFailure> {
        let prepared =
            PreparedSnapshot::capture_for_device(commands, Some(self.packet_key.device))?;
        let payload = if queued {
            Some(self.descriptors.runtime.reserve_payload(
                crate::gpu_runtime::PayloadKind::Frame,
                prepared.packet_bytes(),
            )?)
        } else {
            None
        };
        self.packet_key.packet = self.packet_key.packet.checked_add(1).ok_or_else(|| {
            RenderFailure::new(
                RenderFailureKind::Internal,
                "stage packet sequence exhausted",
            )
        })?;
        self.packet_key.reservation = self.target.frame_reservation();
        self.packet_key.operation = self.active_frame.operations.applied();
        let draw = Arc::new(PacketDraw {
            clear,
            snapshot: prepared.snapshot,
            route,
        });
        self.pending_stage = Some(PendingStage {
            key: self.packet_key,
            draw: Arc::downgrade(&draw),
            content: StageContent::Live(prepared.bindings),
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
        self.descriptors.runtime.check_health()?;
        if self.pending_stage.as_ref().map(|p| p.key) != Some(packet.key)
            || self.target.frame_reservation() != packet.key.reservation
        {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "obsolete stage packet",
            ));
        }
        let frame = self.take_stage_output()?;
        self.submit_packet_to(frame, packet)
    }

    fn submit_packet_to(
        &mut self,
        frame: T::Frame,
        packet: FramePacket,
    ) -> Result<ruffle_render::backend::FrameSubmission, RenderFailure> {
        let mut payload = packet.payload;
        if let Some(ticket) = &mut payload {
            ticket.begin_encoding();
        }
        let pending = self.pending_stage.take().expect("matched pending stage");
        match pending.content {
            StageContent::Failed(error) => return Err(error),
            StageContent::Live(bindings) => {
                let draw = Arc::unwrap_or_clone(packet.draw);
                let clear = draw.clear;
                let commands = self.plan_stage_packet(draw, bindings)?;
                self.encode_stage_draw(frame.view(), clear, commands)?;
            }
            StageContent::Encoded(saved) => {
                run_copy_pipeline(
                    &self.descriptors,
                    self.target.format(),
                    frame.view(),
                    &saved.1,
                    1,
                    self.active_frame.encoder(),
                )?;
            }
        }
        self.finish_stage(frame)
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
        let prepared = PreparedSnapshot {
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

        self.save_pending_stage(draw, false)
    }

    fn save_pending_stage(
        &mut self,
        draw: Arc<PacketDraw>,
        submit_before_return: bool,
    ) -> Result<(), RenderFailure> {
        // Encode V before the write to V+1. This also preserves bitmap fills
        // already baked into resident shape bind groups, without CPU readback.
        let mut pending = self.pending_stage.take().expect("live stage");
        let StageContent::Live(bindings) = pending.content else {
            unreachable!()
        };
        let result = (|| {
            let clear = draw.clear;
            let commands = self.plan_stage_packet(Arc::unwrap_or_clone(draw), bindings)?;
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
            self.encode_stage_draw(&saved.1, clear, commands)?;
            if submit_before_return {
                self.active_frame.submit_direct(&self.descriptors)?;
            }
            Ok::<_, RenderFailure>(saved)
        })();
        let status = result.as_ref().map(|_| ()).map_err(Clone::clone);
        pending.content = match result {
            Ok(saved) => StageContent::Encoded(saved),
            Err(error) => StageContent::Failed(error),
        };
        self.pending_stage = Some(pending);
        status
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
        let display = self
            .display_enhancer
            .as_ref()
            .filter(|e| e.enabled())
            .cloned();
        self.surface.display = display.clone();
        for entry in cache_entries {
            let prepared = crate::render_snapshot::PreparedSnapshot::capture(&entry.commands)?;
            let cache =
                (&*entry.handle.0 as &dyn Any).downcast_ref::<crate::bitmap_cache::CacheImage>();
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
                    self.surface.quality(),
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
                    self.surface.quality(),
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
                self.surface.quality(),
                input.width,
                input.height,
                texture.texture.format(),
            )?
            .with_origin([input.x, input.y])
            .with_display(display.clone());
            let graph = crate::graph_encoder::GraphEncoder::from_prepared(
                self,
                prepared,
                input,
                surface.sample_count(),
                crate::pass_graph::Load::Clear(entry.clear),
            )?;
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
            let mode = if partial {
                RenderTargetMode::FreshWithColor(color)
            } else {
                RenderTargetMode::ExistingWithColor(texture.texture.clone(), color)
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
            if !entry.filters.is_empty()
                && let Some(result) = self.descriptors.filters.apply_chain(
                    &self.descriptors,
                    &mut scope.scope("Filter chain"),
                    &mut self.offscreen_texture_pool,
                    &mut self.active_frame.staging_belt,
                    FilterSource::for_entire_texture(target.color_texture()),
                    entry.filters,
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
            } else if filtered {
                run_copy_pipeline(
                    &self.descriptors,
                    texture.texture.format(),
                    &texture.texture.create_view(&Default::default())?,
                    target.color_view(),
                    target.color_texture().sample_count(),
                    &mut scope.scope("Copy filtered to CAB"),
                )?;
            }
            drop(scope);
            self.finish_bitmap_write(&entry.handle)?;
            if let (Some(tiles), Some(input), Some(cache)) = (&mut self.filtered_tiles, tile, cache)
            {
                if let Some(id) = tiles.store(
                    input,
                    &texture.texture,
                    &self.descriptors.device,
                    self.active_frame.command_encoder.record(),
                )? {
                    cache
                        .tile_id
                        .store(id, std::sync::atomic::Ordering::Relaxed);
                }
            }
            if let Some(cache) = cache {
                cache.finish_redraw(history);
            }
            self.active_frame.maybe_flush(&self.descriptors)?;
        }

        Ok(())
    }

    fn encode_stage_draw(
        &mut self,
        view: &crate::gpu_resources::TextureView,
        clear: Color,
        graph: crate::graph_encoder::SurfaceCommands,
    ) -> Result<(), RenderFailure> {
        self.surface.draw_graph_and_copy_to(
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
            &mut self.profiler.scope(
                ruffle_render::profile::next_frame()
                    .map(|n| format!("Stage {n}"))
                    .unwrap_or_else(|| "Frame commands".into()),
                self.active_frame.command_encoder.record(),
            ),
            &self.meshes,
            graph,
            LayerRef::None,
            &mut self.texture_pool,
        )?;
        Ok(())
    }

    fn finish_stage(
        &mut self,
        frame_output: T::Frame,
    ) -> Result<ruffle_render::backend::FrameSubmission, RenderFailure> {
        if let Some(e) = &self.surface.display {
            e.finish_frame(self.active_frame.command_encoder.record())?;
        }
        self.profiler
            .resolve_queries(self.active_frame.command_encoder.record());
        self.active_frame.staging_belt.finish();

        self.last_submission = Some(self.active_frame.submit_for_target(
            &self.descriptors,
            &self.target,
            frame_output,
        )?);
        if let Some(e) = &self.surface.display {
            e.submitted(self.descriptors.queue.raw());
        }
        let (idle, peak, combined) = self.budget.stats();
        ruffle_render::profile::pool(idle, peak, combined);
        self.texture_pool.end_frame();
        self.offscreen_texture_pool.end_frame();
        self.profiler
            .end_frame()
            .map_err(|error| self.descriptors.runtime.health().record(error.into()))?;
        ruffle_render::profile::submitted();
        self.collect_gpu_samples();

        self.descriptors.runtime.check_health()?;
        Ok(ruffle_render::backend::FrameSubmission::Submitted)
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
        let graph = if self.owned_frames {
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
                    let packet = self.capture_stage_commands(clear, &commands, route, false)?;
                    return self.submit_packet_to(frame_output, packet);
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
        self.encode_stage_draw(frame_output.view(), clear, graph)?;
        self.finish_stage(frame_output)
    }
}
