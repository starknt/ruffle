//! Isolated native replay diagnostics. Pass timestamps use the device clock. A hardware
//! resolve belongs to the raster interval; it has no independent shader invocation count.
use super::{Device, Indexed, Resource};
use fnv::FnvHashMap;
use parking_lot::Mutex;
use std::{
    cell::RefCell,
    collections::VecDeque,
    hash::Hash,
    sync::{Arc, OnceLock, Weak},
};

#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PassDiagnostic {
    pub frame_id: Option<u64>,
    pub kind: &'static str,
    pub pass: &'static str,
    pub filter: Option<&'static str>,
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
    pub resolve: bool,
    pub start_ns: Option<u64>,
    pub end_ns: Option<u64>,
    pub fragment_invocations: Option<u64>,
    pub coverage_pixels: Option<f64>,
    pub method: &'static str,
    pub status: &'static str,
    /// Completed rows evicted before the host drained this device.
    #[serde(skip_serializing_if = "is_zero")]
    pub dropped_rows: u64,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Statistics,
    Coverage,
}
pub(crate) fn requested() -> Option<Mode> {
    match std::env::var("RUFFLE_GPU_PASS_DIAGNOSTICS").as_deref() {
        Ok("1") => Some(Mode::Statistics),
        Ok("coverage") => Some(Mode::Coverage),
        _ => None,
    }
}

#[derive(Default)]
pub(crate) enum Context {
    #[default]
    Outside,
    Stage,
    Target(wgpu::Texture),
    Offscreen,
    Filter(&'static str),
}
thread_local! { static CONTEXT: RefCell<Context> = const { RefCell::new(Context::Outside) }; }
pub(crate) struct Scope(Context);
impl Drop for Scope {
    fn drop(&mut self) {
        CONTEXT.replace(std::mem::take(&mut self.0));
    }
}
impl Device {
    pub(crate) fn pass_context(&self, context: Context) -> Option<Scope> {
        self.0
            .inner()
            .diagnostics
            .as_ref()
            .map(|_| Scope(CONTEXT.replace(context)))
    }
    pub(crate) fn raster_context(&self, target: &wgpu::Texture) -> Option<Scope> {
        self.0.inner().diagnostics.as_ref()?;
        let context = CONTEXT.with_borrow(|c| match c {
            Context::Stage => Context::Target(target.clone()),
            _ => Context::Offscreen,
        });
        self.pass_context(context)
    }
    pub fn take_pass_diagnostics(&self) -> Vec<PassDiagnostic> {
        self.0
            .inner()
            .diagnostics
            .as_ref()
            .map_or_else(Vec::new, |d| d.ready.lock().drain(..).collect())
    }
    pub(crate) fn configure_coverage(&self, adapter: &wgpu::Adapter) {
        if let Some(d) = &self.0.inner().diagnostics
            && d.mode == Mode::Coverage
        {
            let supported = adapter
                .get_downlevel_capabilities()
                .flags
                .contains(wgpu::DownlevelFlags::INDEPENDENT_BLEND)
                && self.limits().max_color_attachments >= 2;
            let flags = if !supported {
                wgpu::TextureFormatFeatureFlags::empty()
            } else if self
                .features()
                .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
            {
                adapter.get_texture_format_features(FORMAT).flags
            } else {
                FORMAT.guaranteed_format_features(self.features()).flags
            };
            let _ = d.format.set(flags);
        }
    }
}

// A key keeps the raw identity unique until its tracked owner drops. Prune dead
// entries on insertion so replaying many different movies cannot grow the cache.
type Variants<T, V = T> = Mutex<FnvHashMap<T, (Weak<dyn Indexed>, V)>>;
fn remember<T: Clone + Eq + Hash + Send + Sync + 'static, V>(
    map: &Variants<T, V>,
    key: &Resource<T>,
    variant: V,
) {
    let mut map = map.lock();
    map.retain(|_, (owner, _)| owner.strong_count() != 0);
    let owner: Arc<dyn Indexed> = key.0.clone();
    map.insert(key.raw().clone(), (Arc::downgrade(&owner), variant));
}
pub(super) struct Diagnostics {
    mode: Mode,
    format: OnceLock<wgpu::TextureFormatFeatureFlags>,
    shaders: Variants<wgpu::ShaderModule>,
    pipelines: Variants<wgpu::RenderPipeline, (wgpu::RenderPipeline, bool)>,
    unsupported: Mutex<std::collections::HashSet<(wgpu::TextureFormat, u32)>>,
    ready: Mutex<VecDeque<PassDiagnostic>>,
    pending: Mutex<(usize, u64)>,
}
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const MAX_ROWS: usize = 8192;
const MAX_BYTES: u64 = 128 << 20;
impl Diagnostics {
    pub(super) fn new(device: &wgpu::Device) -> Option<Arc<Self>> {
        let mode = match requested()? {
            Mode::Statistics
                if device
                    .features()
                    .contains(wgpu::Features::PIPELINE_STATISTICS_QUERY) =>
            {
                Mode::Statistics
            }
            _ => Mode::Coverage,
        };
        Some(Arc::new(Self {
            mode,
            format: OnceLock::new(),
            shaders: Default::default(),
            pipelines: Default::default(),
            unsupported: Default::default(),
            ready: Default::default(),
            pending: Default::default(),
        }))
    }
    pub(super) fn shader(
        &self,
        device: &Device,
        desc: &wgpu::ShaderModuleDescriptor<'_>,
        original: &super::ShaderModule,
    ) {
        if self.mode == Mode::Coverage
            && let Some(variant) = super::coverage_shader::shader(device, desc)
        {
            remember(&self.shaders, original, variant);
        }
    }
    fn coverage_supported(&self, samples: u32) -> bool {
        self.mode == Mode::Coverage
            && self.format.get().is_some_and(|flags| {
                flags.contains(wgpu::TextureFormatFeatureFlags::BLENDABLE)
                    && flags.sample_count_supported(samples)
                    && (samples == 1
                        || flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE))
            })
    }
    pub(super) fn pipeline(
        &self,
        device: &Device,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
        original: &super::RenderPipeline,
    ) {
        if !self.coverage_supported(desc.multisample.count) {
            return;
        }
        let Some(fragment) = &desc.fragment else {
            return;
        };
        let [Some(color)] = fragment.targets else {
            return;
        };
        let shader = self
            .shaders
            .lock()
            .get(fragment.module)
            .map(|(_, shader)| shader.clone());
        let add = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        // A shader with unsupported outputs still renders its original color. Bind
        // a write-disabled companion and mark that whole pass's coverage missing.
        let build = |shader: Option<&wgpu::ShaderModule>| {
            let active = shader.is_some();
            let targets = [
                Some(color.clone()),
                Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: active.then_some(wgpu::BlendState {
                        color: add,
                        alpha: add,
                    }),
                    write_mask: if active {
                        wgpu::ColorWrites::RED
                    } else {
                        wgpu::ColorWrites::empty()
                    },
                }),
            ];
            let desc = wgpu::RenderPipelineDescriptor {
                fragment: Some(wgpu::FragmentState {
                    module: shader.unwrap_or(fragment.module),
                    targets: &targets,
                    ..fragment.clone()
                }),
                ..desc.clone()
            };
            device
                .scoped(|d| Ok(d.create_render_pipeline(&desc)))
                .ok()
                .map(|pipeline| (pipeline, active))
        };
        let variant = desc.layout.and_then(|_| {
            shader
                .as_ref()
                .and_then(|s| build(Some(s)))
                .or_else(|| build(None))
        });
        if let Some(variant) = variant {
            remember(&self.pipelines, original, variant);
        } else {
            self.unsupported
                .lock()
                .insert((color.format, desc.multisample.count));
        }
    }
    pub(super) fn variant(
        &self,
        pipeline: &wgpu::RenderPipeline,
    ) -> Option<(wgpu::RenderPipeline, bool)> {
        self.pipelines.lock().get(pipeline).map(|(_, p)| p.clone())
    }
    fn publish(&self, mut row: PassDiagnostic) {
        let mut rows = self.ready.lock();
        if rows.len() == MAX_ROWS
            && let Some(old) = rows.pop_front()
        {
            row.dropped_rows = old.dropped_rows.saturating_add(1);
        }
        rows.push_back(row);
    }
}

struct Reservation {
    sink: Arc<Diagnostics>,
    bytes: u64,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        let mut pending = self.sink.pending.lock();
        pending.0 -= 1;
        pending.1 -= self.bytes;
    }
}
pub(super) struct Coverage {
    pub(super) view: wgpu::TextureView,
    pub(super) resolve: Option<wgpu::TextureView>,
    image: wgpu::Texture,
    stride: u32,
}
impl Coverage {
    fn sum(&self, bytes: &[u8], width: u32, samples: u32) -> Option<f64> {
        let mut total = 0.;
        for line in bytes.chunks_exact(self.stride as usize) {
            for pixel in line[..width as usize * 2].as_chunks::<2>().0 {
                let bits = u16::from_le_bytes(*pixel);
                if bits == 0 {
                    continue;
                }
                // Values are positive normal binary16 (the smallest is 1/16).
                let value = f32::from_bits((u32::from(bits) << 13) + ((127 - 15) << 23));
                // A resolved pixel may hide a saturated sample. Above 2048,
                // adding one is no longer exact in binary16.
                if value * samples as f32 >= 2048. {
                    return None;
                }
                total += f64::from(value);
            }
        }
        Some(total)
    }
    fn new(device: &wgpu::Device, row: &PassDiagnostic, stride: u32) -> Self {
        let size = wgpu::Extent3d {
            width: row.width,
            height: row.height,
            depth_or_array_layers: 1,
        };
        let texture = |sample_count| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Additive fragment coverage"),
                size,
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: if sample_count == 1 {
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
                } else {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                },
                view_formats: &[],
            })
        };
        let image = texture(1);
        let single = image.create_view(&Default::default());
        let (view, resolve) = if row.sample_count == 1 {
            (single, None)
        } else {
            (
                texture(row.sample_count).create_view(&Default::default()),
                Some(single),
            )
        };
        Self {
            view,
            resolve,
            image,
            stride,
        }
    }
}
pub(super) struct Pending {
    pub(super) row: PassDiagnostic,
    reservation: Option<Reservation>,
    sink: Arc<Diagnostics>,
    pub(super) time: Option<(wgpu::QuerySet, u32, u32)>,
    pub(super) statistics: Option<wgpu::QuerySet>,
    pub(super) coverage: Option<Coverage>,
}
impl Pending {
    pub(super) fn raster(device: &Device, desc: &wgpu::RenderPassDescriptor<'_>) -> Option<Self> {
        let a = desc.color_attachments.first()?.as_ref()?;
        let texture = a.view.texture();
        let writes = desc.timestamp_writes.as_ref().and_then(|w| {
            Some((
                w.query_set.clone(),
                w.beginning_of_pass_write_index?,
                w.end_of_pass_write_index?,
            ))
        });
        Self::new(
            device,
            Some(texture),
            (texture.width(), texture.height()),
            a.resolve_target.is_some(),
            writes,
            desc.timestamp_writes.is_none(),
        )
    }
    pub(super) fn compute(
        device: &Device,
        size: (u32, u32),
        desc: &wgpu::ComputePassDescriptor<'_>,
    ) -> Option<Self> {
        let writes = desc.timestamp_writes.as_ref().and_then(|w| {
            Some((
                w.query_set.clone(),
                w.beginning_of_pass_write_index?,
                w.end_of_pass_write_index?,
            ))
        });
        Self::new(
            device,
            None,
            size,
            false,
            writes,
            desc.timestamp_writes.is_none(),
        )
    }
    fn new(
        device: &Device,
        raster: Option<&wgpu::Texture>,
        (width, height): (u32, u32),
        resolve: bool,
        writes: Option<(wgpu::QuerySet, u32, u32)>,
        install_timestamps: bool,
    ) -> Option<Self> {
        let sink = device.0.inner().diagnostics.as_ref()?;
        let (kind, filter) = CONTEXT.with_borrow(|c| match c {
            Context::Outside => None,
            Context::Filter(name) => Some(("filter", Some(*name))),
            Context::Target(target) if Some(target) == raster => Some(("stage", None)),
            Context::Stage => Some(("stage", None)),
            _ => Some(("offscreen", None)),
        })?;
        let sample_count = raster.map_or(1, wgpu::Texture::sample_count);
        let statistics = raster.is_some() && sink.mode == Mode::Statistics;
        let coverage = sink.coverage_supported(sample_count)
            && raster.is_some_and(|t| {
                !sink
                    .unsupported
                    .lock()
                    .contains(&(t.format(), sample_count))
            });
        let mut row = PassDiagnostic {
            frame_id: ruffle_render::profile::next_frame(),
            kind,
            pass: if raster.is_some() {
                "raster"
            } else {
                "compute"
            },
            filter,
            width,
            height,
            sample_count,
            resolve,
            start_ns: None,
            end_ns: None,
            fragment_invocations: None,
            coverage_pixels: None,
            method: if statistics {
                "pipeline-statistics"
            } else if raster.is_some() {
                "additive-coverage"
            } else {
                "timestamp"
            },
            status: "ok",
            dropped_rows: 0,
        };
        if raster.is_some() && !statistics && !coverage {
            row.status = "coverage-unavailable";
        }
        let stride = (width * 2).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let bytes = 1536
            + if coverage {
                u64::from(height)
                    * (u64::from(stride)
                        + u64::from(width)
                            * 2
                            * u64::from(sample_count + u32::from(sample_count > 1)))
            } else {
                0
            };
        {
            let mut pending = sink.pending.lock();
            if pending.0 >= 1024 || pending.1 + bytes > MAX_BYTES {
                row.status = "readback-budget";
                return Some(Self {
                    row,
                    time: None,
                    coverage: None,
                    statistics: None,
                    reservation: None,
                    sink: sink.clone(),
                });
            }
            pending.0 += 1;
            pending.1 += bytes;
        }
        let reservation = Reservation {
            sink: sink.clone(),
            bytes,
        };
        let query = |ty, count| {
            device.raw().create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Native pass diagnostic"),
                ty,
                count,
            })
        };
        let time = writes.or_else(|| {
            (install_timestamps && device.features().contains(wgpu::Features::TIMESTAMP_QUERY))
                .then(|| (query(wgpu::QueryType::Timestamp, 2), 0, 1))
        });
        let statistics = statistics.then(|| {
            query(
                wgpu::QueryType::PipelineStatistics(
                    wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS,
                ),
                1,
            )
        });
        let coverage = coverage.then(|| Coverage::new(device.raw(), &row, stride));
        Some(Self {
            row,
            time,
            statistics,
            coverage,
            reservation: Some(reservation),
            sink: sink.clone(),
        })
    }
    pub(super) fn finish(mut self, device: &Device, encoder: &mut wgpu::CommandEncoder) {
        // Resolve offsets must be 256-byte aligned, even for disjoint profiler indices.
        const STRIDE: u64 = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;
        const HEADER: u64 = STRIDE * 3;
        if self.time.is_none() && self.coverage.is_none() && self.statistics.is_none() {
            encoder.on_submitted_work_done(move || self.sink.publish(self.row));
            return;
        }
        let buffer = |size, usage| {
            device.raw().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Native pass diagnostic"),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let size = HEADER
            + self
                .coverage
                .as_ref()
                .map_or(0, |c| u64::from(c.stride) * u64::from(self.row.height));
        let read = buffer(
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        if self.time.is_some() || self.statistics.is_some() {
            let resolved = buffer(
                HEADER,
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            );
            if let Some((set, start, end)) = &self.time {
                encoder.resolve_query_set(set, *start..start + 1, &resolved, 0);
                encoder.resolve_query_set(set, *end..end + 1, &resolved, STRIDE);
            }
            if let Some(stats) = &self.statistics {
                encoder.resolve_query_set(stats, 0..1, &resolved, STRIDE * 2);
            }
            encoder.copy_buffer_to_buffer(&resolved, 0, &read, 0, HEADER);
        }
        if let Some(c) = &self.coverage {
            encoder.copy_texture_to_buffer(
                c.image.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &read,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: HEADER,
                        bytes_per_row: Some(c.stride),
                        rows_per_image: None,
                    },
                },
                c.image.size(),
            );
        }
        let period = device.0.inner().queue.get_timestamp_period();
        let mapped = read.clone();
        encoder.map_buffer_on_submit(&read, wgpu::MapMode::Read, .., move |result| {
            if let Ok(bytes) = result.and_then(|()| {
                mapped
                    .slice(..)
                    .get_mapped_range()
                    .map_err(|_| wgpu::BufferAsyncError)
            }) {
                let word = |offset: u64| {
                    u64::from_le_bytes(
                        bytes[offset as usize..offset as usize + 8]
                            .try_into()
                            .unwrap(),
                    )
                };
                if self.time.is_some()
                    && let Some(ns) =
                        crate::gpu_profiler::time_range_ns(word(0)..word(STRIDE), period)
                {
                    self.row.start_ns = Some(ns.start);
                    self.row.end_ns = Some(ns.end);
                }
                if self.statistics.is_some() {
                    self.row.fragment_invocations = Some(word(STRIDE * 2));
                }
                if let Some(c) = self.coverage
                    && self.row.status == "ok"
                {
                    self.row.coverage_pixels = c.sum(
                        &bytes[HEADER as usize..],
                        self.row.width,
                        self.row.sample_count,
                    );
                    if self.row.coverage_pixels.is_none() {
                        self.row.status = "coverage-saturated";
                    }
                }
                drop(bytes);
                mapped.unmap();
            } else {
                self.row.status = "map-failed";
            }
            self.sink.publish(self.row);
            drop(self.reservation);
        });
    }
}
