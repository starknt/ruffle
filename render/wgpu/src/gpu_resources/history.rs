//! Only structural recipes leave the process. Shader modules and binding layouts must be
//! supplied by this device, from the running renderer or a program the movie has uploaded.
use super::*;
use crate::context3d::shader_pair::{AgalProgram, ShaderCompileData};
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Read, Write},
    num::NonZeroU32,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
};

const SCHEMA: u32 = 4;
const MAX_ENTRIES: usize = 128;
const MAX_JOBS: usize = 16;
const MAX_FILE: u64 = 1024 * 1024;
const MAX_FILES: usize = 64;
type Job = Box<dyn FnOnce() + Send>;

thread_local! { static SPECULATIVE:std::cell::Cell<bool>=const {std::cell::Cell::new(false)}; }
pub(super) fn speculative() -> bool {
    SPECULATIVE.get()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Layout {
    pub groups: Vec<Option<Signature>>,
    pub immediate: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Stage {
    module: Signature,
    entry: Option<String>,
    constants: Vec<(String, u64)>,
    zero: bool,
}
impl Stage {
    fn capture(
        device: &Device,
        module: &wgpu::ShaderModule,
        entry: Option<&str>,
        opts: &wgpu::PipelineCompilationOptions<'_>,
    ) -> Option<Self> {
        Some(Self {
            module: device.resource_signature(module)?,
            entry: entry.map(str::to_owned),
            constants: opts
                .constants
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_bits()))
                .collect(),
            zero: opts.zero_initialize_workgroup_memory,
        })
    }
    fn constants(&self) -> Vec<(&str, f64)> {
        self.constants
            .iter()
            .map(|(n, v)| (n.as_str(), f64::from_bits(*v)))
            .collect()
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct VertexStream {
    stride: u64,
    step: wgpu::VertexStepMode,
    attributes: Vec<wgpu::VertexAttribute>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
enum Pipeline {
    Render {
        vertex: Stage,
        buffers: Vec<Option<VertexStream>>,
        fragment: Option<(Stage, Vec<Option<wgpu::ColorTargetState>>)>,
        primitive: wgpu::PrimitiveState,
        depth: Option<wgpu::DepthStencilState>,
        samples: wgpu::MultisampleState,
        views: Option<NonZeroU32>,
    },
    Compute(Stage),
    Ready(Signature),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Recipe {
    family: Family,
    pub(super) agal: Option<(Signature, ShaderCompileData)>,
    layout: Option<Layout>,
    pipeline: Pipeline,
}
impl Recipe {
    fn layout(d: &Device, layout: Option<&wgpu::PipelineLayout>) -> Option<Option<Layout>> {
        if let Some(l) = layout {
            Some(Some(d.lookup_object(l).ok()?.layout()?.clone()))
        } else {
            Some(None)
        }
    }
    pub fn render(
        d: &Device,
        family: Family,
        v: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> Option<Self> {
        let vertex = Stage::capture(
            d,
            v.vertex.module,
            v.vertex.entry_point,
            &v.vertex.compilation_options,
        )?;
        let fragment = if let Some(f) = &v.fragment {
            Some((
                Stage::capture(d, f.module, f.entry_point, &f.compilation_options)?,
                f.targets.to_vec(),
            ))
        } else {
            None
        };
        Some(Self {
            family,
            agal: None,
            layout: Self::layout(d, v.layout)?,
            pipeline: Pipeline::Render {
                vertex,
                fragment,
                buffers: v
                    .vertex
                    .buffers
                    .iter()
                    .map(|b| {
                        b.as_ref().map(|b| VertexStream {
                            stride: b.array_stride,
                            step: b.step_mode,
                            attributes: b.attributes.to_vec(),
                        })
                    })
                    .collect(),
                primitive: v.primitive,
                depth: v.depth_stencil.clone(),
                samples: v.multisample,
                views: v.multiview_mask,
            },
        })
    }
    pub fn compute(
        d: &Device,
        family: Family,
        v: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> Option<Self> {
        Some(Self {
            family,
            agal: None,
            layout: Self::layout(d, v.layout)?,
            pipeline: Pipeline::Compute(Stage::capture(
                d,
                v.module,
                v.entry_point,
                &v.compilation_options,
            )?),
        })
    }
    pub fn ready(family: Family, key: Signature) -> Self {
        Self {
            family,
            agal: None,
            layout: None,
            pipeline: Pipeline::Ready(key),
        }
    }
    fn key(&self) -> Signature {
        signature(format_args!("{self:?}"))
    }
    fn resolve(&self, sources: &Sources) -> Option<Available> {
        if let Some((content, data)) = &self.agal {
            return Some(Available::Agal(
                sources.agal.get(content)?.upgrade()?,
                data.clone(),
            ));
        }

        let kernel_stage = match &self.pipeline {
            Pipeline::Compute(stage) => Some(stage),
            Pipeline::Render {
                vertex,
                fragment: Some((fragment, _)),
                ..
            } if vertex.module == fragment.module => Some(vertex),
            _ => None,
        };
        if self.family == Family::Upscale
            && let Some(stage) = kernel_stage
            && let Some(layout) = &self.layout
            && let [Some(group)] = layout.groups.as_slice()
            && let Some(kernel) = sources.kernels.get(&(stage.module, *group))
        {
            return Some(Available::Kernel(kernel.clone()));
        }
        let module = |key| sources.shaders.get(&key)?.upgrade().map(Resource);
        let (vertex, fragment) = match &self.pipeline {
            Pipeline::Render {
                vertex, fragment, ..
            } => (
                module(vertex.module)?,
                if let Some((f, _)) = fragment {
                    Some(module(f.module)?)
                } else {
                    None
                },
            ),
            Pipeline::Compute(stage) => (module(stage.module)?, None),
            Pipeline::Ready(_) => return None,
        };
        let groups = self.layout.as_ref().map_or(Some(vec![]), |l| {
            l.groups
                .iter()
                .map(|g| match g {
                    Some(k) => Some(Some(Resource(sources.groups.get(k)?.upgrade()?))),
                    None => Some(None),
                })
                .collect()
        })?;
        Some(Available::Native(Inputs {
            vertex,
            fragment,
            groups,
        }))
    }
    fn compile(&self, d: &Device, inputs: &Inputs) -> Result<Compiled, RenderFailure> {
        let groups: Vec<_> = inputs
            .groups
            .iter()
            .map(|g| g.as_ref().map(Resource::raw))
            .collect();
        let layout = self
            .layout
            .as_ref()
            .map(|l| {
                d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("History pipeline layout"),
                    bind_group_layouts: &groups,
                    immediate_size: l.immediate,
                })
            })
            .transpose()?;
        match &self.pipeline {
            Pipeline::Render {
                vertex,
                buffers,
                fragment,
                primitive,
                depth,
                samples,
                views,
            } => {
                let buffers: Vec<_> = buffers
                    .iter()
                    .map(|v| {
                        v.as_ref().map(|v| wgpu::VertexBufferLayout {
                            array_stride: v.stride,
                            step_mode: v.step,
                            attributes: &v.attributes,
                        })
                    })
                    .collect();
                let vc = vertex.constants();
                let fc = fragment
                    .as_ref()
                    .map(|(f, _)| f.constants())
                    .unwrap_or_default();
                let desc = wgpu::RenderPipelineDescriptor {
                    label: Some("History render pipeline"),
                    layout: layout.as_ref().map(Resource::raw),
                    vertex: wgpu::VertexState {
                        module: &inputs.vertex,
                        entry_point: vertex.entry.as_deref(),
                        buffers: &buffers,
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &vc,
                            zero_initialize_workgroup_memory: vertex.zero,
                        },
                    },
                    fragment: fragment.as_ref().zip(inputs.fragment.as_ref()).map(
                        |((f, targets), module)| wgpu::FragmentState {
                            module,
                            entry_point: f.entry.as_deref(),
                            targets,
                            compilation_options: wgpu::PipelineCompilationOptions {
                                constants: &fc,
                                zero_initialize_workgroup_memory: f.zero,
                            },
                        },
                    ),
                    primitive: *primitive,
                    depth_stencil: depth.clone(),
                    multisample: *samples,
                    multiview_mask: *views,
                    cache: d.pipeline_cache(),
                };
                let p = d.create_render_direct(self.family, &desc)?;
                Ok(Compiled::Render(
                    p.raw().clone(),
                    p.lease().clone(),
                    p.0.metadata.get().and_then(|m| m.profile.get()).cloned(),
                ))
            }
            Pipeline::Compute(stage) => {
                let constants = stage.constants();
                let desc = wgpu::ComputePipelineDescriptor {
                    label: Some("History compute pipeline"),
                    layout: layout.as_ref().map(Resource::raw),
                    module: &inputs.vertex,
                    entry_point: stage.entry.as_deref(),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        zero_initialize_workgroup_memory: stage.zero,
                    },
                    cache: d.pipeline_cache(),
                };
                let p = d.create_compute_direct(self.family, &desc)?;
                Ok(Compiled::Compute(
                    p.raw().clone(),
                    p.lease().clone(),
                    p.0.metadata.get().and_then(|m| m.profile.get()).cloned(),
                ))
            }
            Pipeline::Ready(_) => unreachable!(),
        }
    }
}
#[derive(Clone)]
struct Kernel {
    name: &'static str,
    source: &'static str,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
}
enum Available {
    Kernel(Arc<Kernel>),
    Native(Inputs),
    Agal(Arc<AgalProgram>, ShaderCompileData),
}
impl Available {
    fn compile(&self, d: &Device, recipe: &Recipe) -> Result<Compiled, RenderFailure> {
        match self {
            Self::Native(inputs) => recipe.compile(d, inputs),
            Self::Kernel(k) => {
                let shader = d.create_shader_module_for(
                    Family::Upscale,
                    wgpu::ShaderModuleDescriptor {
                        label: Some(k.name),
                        source: wgpu::ShaderSource::Wgsl(k.source.into()),
                    },
                )?;
                let group = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(k.name),
                    entries: &k.entries,
                })?;
                recipe.compile(
                    d,
                    &Inputs {
                        fragment: matches!(recipe.pipeline, Pipeline::Render { .. })
                            .then(|| shader.clone()),
                        vertex: shader,
                        groups: vec![Some(group)],
                    },
                )
            }
            Self::Agal(program, spec) => {
                let shaders = program.compile(d, spec.clone()).map_err(|e| {
                    RenderFailure::new(RenderFailureKind::Internal, format!("AGAL prewarm: {e:?}"))
                })?;
                let Pipeline::Render {
                    vertex,
                    fragment: Some((fragment, _)),
                    ..
                } = &recipe.pipeline
                else {
                    return Err(RenderFailure::new(
                        RenderFailureKind::Validation,
                        "invalid AGAL recipe",
                    ));
                };
                if d.resource_signature(shaders.vertex_module.raw()) != Some(vertex.module)
                    || d.resource_signature(shaders.fragment_module.raw()) != Some(fragment.module)
                {
                    return Err(RenderFailure::new(
                        RenderFailureKind::Validation,
                        "AGAL specialization changed",
                    ));
                }
                recipe.compile(
                    d,
                    &Inputs {
                        vertex: shaders.vertex_module.clone(),
                        fragment: Some(shaders.fragment_module.clone()),
                        groups: vec![Some(shaders.bind_group_layout.clone())],
                    },
                )
            }
        }
    }
}
struct Inputs {
    vertex: ShaderModule,
    fragment: Option<ShaderModule>,
    groups: Vec<Option<BindGroupLayout>>,
}
pub(super) enum Compiled {
    Render(wgpu::RenderPipeline, ResourceLease, Option<Arc<ObjectInfo>>),
    Compute(
        wgpu::ComputePipeline,
        ResourceLease,
        Option<Arc<ObjectInfo>>,
    ),
}
#[derive(Default)]
struct Sources {
    kernels: BTreeMap<(Signature, Signature), Arc<Kernel>>,
    agal: BTreeMap<Signature, Weak<AgalProgram>>,
    shaders: BTreeMap<Signature, Weak<Object<wgpu::ShaderModule>>>,
    groups: BTreeMap<Signature, Weak<Object<wgpu::BindGroupLayout>>>,
}
fn register<T>(
    map: &mut BTreeMap<Signature, Weak<Object<T>>>,
    key: Signature,
    resource: &Resource<T>,
) {
    if map.len() >= 512 {
        map.retain(|_, v| v.strong_count() > 0);
    }
    if map.len() < 512 || map.contains_key(&key) {
        map.insert(key, Arc::downgrade(&resource.0));
    }
}
struct Slot {
    age: u64,
    creation_ns: u64,
    canceled: AtomicBool,
    ticket: OnceCell<crate::resource_scheduler::CacheTicket>,
    recipe: Recipe,
    result: OnceCell<Compiled>,
    queued: AtomicBool,
    warm: AtomicBool,
}
impl Slot {
    fn new(recipe: Recipe, age: u64, creation_ns: u64) -> Self {
        Self {
            age,
            creation_ns,
            canceled: false.into(),
            ticket: OnceCell::new(),
            recipe,
            result: OnceCell::new(),
            queued: false.into(),
            warm: false.into(),
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
struct Entry {
    #[serde(default)]
    warm: bool,
    recipe: Recipe,
    uses: u64,
    first: u64,
    ns: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Manifest {
    checksum: String,
    schema: u32,
    identity: String,
    entries: Vec<Entry>,
    warm_uses: u64,
}
pub(super) struct Observation {
    recipe: Recipe,
    key: Signature,
    ns: u64,
    warm_epoch: Option<u64>,
}
impl Observation {
    pub fn new(recipe: Recipe, ns: u64, warm_epoch: Option<u64>) -> Arc<Self> {
        Arc::new(Self {
            key: recipe.key(),
            recipe,
            ns,
            warm_epoch,
        })
    }
}
struct Movie {
    content: [u8; 32],
    quality: String,
    identity: String,
    session: u64,
    used: BTreeMap<Signature, Entry>,
    wanted: Vec<(Signature, Arc<Slot>)>,
    warm_uses: u64,
}
struct State {
    presentation: String,
    movie: Option<Movie>,
    sources: Sources,
}
impl crate::resource_scheduler::Reclaim for Mutex<State> {
    fn oldest(&self) -> Option<u64> {
        self.lock()
            .movie
            .as_ref()?
            .wanted
            .first()
            .map(|(_, s)| s.age)
    }
    fn evict_oldest(&self) -> bool {
        let mut state = self.lock();
        let Some(movie) = state.movie.as_mut() else {
            return false;
        };
        if movie.wanted.is_empty() {
            return false;
        }
        let (_, slot) = movie.wanted.remove(0);
        slot.canceled.store(true, Ordering::Release);
        true
    }
}
pub(super) struct History {
    dir: PathBuf,
    abi: String,
    consume: bool,
    state: Arc<Mutex<State>>,
    transition: Mutex<()>,
    epoch: AtomicU64,
    jobs: mpsc::SyncSender<Job>,
}
impl History {
    pub fn open(d: &Device, dir: &Path, abi: &str) -> Option<Self> {
        let mode = std::env::var("RUFFLE_PIPELINE_HISTORY").ok()?;
        if !matches!(mode.as_str(), "1" | "record") {
            return None;
        }
        let dir = crate::pipeline_cache::private_dir(dir).ok()?;
        let (jobs, queue) = mpsc::sync_channel::<Job>(MAX_JOBS);
        std::thread::Builder::new()
            .name("wgpu-pipeline-warm".into())
            .spawn(move || {
                for job in queue {
                    SPECULATIVE.set(true);
                    job();
                    SPECULATIVE.set(false);
                }
            })
            .ok()?;
        let state = Arc::new(Mutex::new(State {
            movie: None,
            sources: Sources::default(),
            presentation: String::new(),
        }));
        d.runtime().scheduler().register(
            crate::resource_scheduler::ReclaimClass::Cache,
            state.clone(),
        );
        Some(Self {
            dir,
            abi: format!(
                "{SCHEMA} {abi} {} {:?} {:?} {:?}",
                crate::pipeline_cache::ENGINE_ABI,
                d.adapter_info(),
                d.features(),
                d.limits()
            ),
            consume: mode == "1",
            state,
            transition: Mutex::new(()),
            epoch: AtomicU64::new(0),
            jobs,
        })
    }
    pub fn kernel(
        &self,
        d: &Device,
        name: &'static str,
        source: &'static str,
        mut entries: Vec<wgpu::BindGroupLayoutEntry>,
    ) {
        entries.sort_unstable_by_key(|e| e.binding);
        let key = (
            signature(format_args!(
                "{:?}",
                wgpu::ShaderSource::Wgsl(source.into())
            )),
            signature(format_args!("{entries:?}")),
        );
        {
            let mut state = self.state.lock();
            if state.sources.kernels.len() < 128 {
                state.sources.kernels.entry(key).or_insert_with(|| {
                    Arc::new(Kernel {
                        name,
                        source,
                        entries,
                    })
                });
            }
        }
        self.schedule(d);
    }
    pub fn shader(&self, d: &Device, key: Signature, module: &ShaderModule) {
        register(&mut self.state.lock().sources.shaders, key, module);
        self.schedule(d);
    }
    pub fn agal(&self, d: &Device, program: &Arc<AgalProgram>) {
        {
            let mut state = self.state.lock();
            state.sources.agal.retain(|_, p| p.strong_count() != 0);
            if state.sources.agal.len() < 512 {
                state
                    .sources
                    .agal
                    .insert(program.content, Arc::downgrade(program));
            }
        }
        self.schedule(d);
    }
    pub fn group(&self, d: &Device, key: Signature, group: &BindGroupLayout) {
        register(&mut self.state.lock().sources.groups, key, group);
        self.schedule(d);
    }
    pub fn presentation(&self, d: &Device, value: String) {
        let _transition = self.transition.lock();
        {
            let mut state = self.state.lock();
            if state.presentation == value {
                return;
            }
            state.presentation = value;
        }
        self.rotate(d, None);
    }
    pub fn quality(&self, d: &Device, session: u64, value: String) {
        let _transition = self.transition.lock();
        if self
            .state
            .lock()
            .movie
            .as_ref()
            .is_some_and(|m| m.session == session && m.quality != value)
        {
            self.rotate(d, Some(value));
        }
    }
    fn rotate(&self, d: &Device, quality: Option<String>) {
        let Some(last) = self.take_movie(None, false) else {
            return;
        };
        let _ = self.write(Self::snapshot(&last));
        self.start(
            d,
            last.session,
            last.content,
            quality.unwrap_or(last.quality),
        );
    }
    pub fn begin(&self, d: &Device, session: u64, content: &[u8], header: &str, quality: &str) {
        let _transition = self.transition.lock();
        if let Some(last) = self.take_movie(None, true) {
            let _ = self.write(Self::snapshot(&last));
        }
        let content = Sha256::new()
            .chain_update(header.as_bytes())
            .chain_update([0])
            .chain_update(content)
            .finalize()
            .into();
        self.start(d, session, content, quality.to_owned());
    }
    fn start(&self, d: &Device, session: u64, content: [u8; 32], quality: String) {
        let identity = format!(
            "{:x}",
            Sha256::new()
                .chain_update(self.abi.as_bytes())
                .chain_update([0])
                .chain_update(self.state.lock().presentation.as_bytes())
                .chain_update([0])
                .chain_update(quality.as_bytes())
                .chain_update([0])
                .chain_update(content)
                .finalize()
        );
        let mut wanted = vec![];
        if self.consume
            && let Some(mut saved) = self.read(&identity)
        {
            saved
                .entries
                .sort_by_key(|e| (e.first, std::cmp::Reverse(e.ns)));
            for e in saved
                .entries
                .into_iter()
                .filter(|e| !matches!(e.recipe.pipeline, Pipeline::Ready(_)))
                .take(MAX_JOBS)
            {
                wanted.push((
                    e.recipe.key(),
                    Arc::new(Slot::new(e.recipe, d.runtime().scheduler().tick(), e.ns)),
                ));
            }
        }
        self.state.lock().movie = Some(Movie {
            content,
            quality,
            identity,
            session,
            used: BTreeMap::new(),
            wanted,
            warm_uses: 0,
        });
        self.schedule(d);
    }
    fn read(&self, identity: &str) -> Option<Manifest> {
        let f =
            crate::pipeline_cache::open_private_read(&self.dir.join(format!("{identity}.recipes")))
                .ok()?;
        if !f.metadata().ok()?.is_file() || f.metadata().ok()?.len() > MAX_FILE {
            return None;
        }
        let mut bytes = vec![];
        f.take(MAX_FILE + 1).read_to_end(&mut bytes).ok()?;
        if bytes.len() as u64 > MAX_FILE {
            return None;
        }
        let m: Manifest = serde_json::from_slice(&bytes).ok()?;
        if m.checksum != Self::checksum(&m.entries)? {
            return None;
        }
        (m.schema == SCHEMA && m.identity == identity && m.entries.len() <= MAX_ENTRIES)
            .then_some(m)
    }
    fn schedule(&self, d: &Device) {
        if speculative() || !self.consume {
            return;
        }
        let state = self.state.lock();
        let Some(movie) = &state.movie else { return };
        let generation = self.epoch.load(Ordering::Acquire);
        for (_, slot) in &movie.wanted {
            if slot.queued.load(Ordering::Relaxed) || slot.result.get().is_some() {
                continue;
            }
            let Some(inputs) = slot.recipe.resolve(&state.sources) else {
                continue;
            };
            slot.queued.store(true, Ordering::Relaxed);
            let task = slot.clone();
            let d = d.clone();
            let sent = self.jobs.try_send(Box::new(move || {
                let Some(history) = d.history() else { return };
                if generation != history.epoch.load(Ordering::Acquire)
                    || task.canceled.load(Ordering::Acquire)
                    || !d.speculation_fits()
                {
                    return;
                }
                let _ = task.result.get_or_try_init(|| {
                    let Some(ticket) = d
                        .runtime()
                        .scheduler()
                        .admit_optional(crate::resource_scheduler::ReclaimClass::Cache, 4096)
                    else {
                        return Err(RenderFailure::new(
                            RenderFailureKind::BudgetExceeded,
                            "prewarm budget",
                        ));
                    };
                    let result = inputs.compile(&d, &task.recipe)?;
                    if generation != history.epoch.load(Ordering::Acquire)
                        || task.canceled.load(Ordering::Acquire)
                    {
                        return Err(RenderFailure::new(
                            RenderFailureKind::Internal,
                            "movie changed",
                        ));
                    }
                    let _ = task.ticket.set(ticket);
                    task.warm.store(true, Ordering::Release);
                    d.schedule_pipeline_cache_save();
                    Ok(result)
                });
            }));
            if sent.is_err() {
                slot.queued.store(false, Ordering::Relaxed);
            }
        }
    }
    fn slot(&self, key: Signature) -> Option<Arc<Slot>> {
        self.state
            .lock()
            .movie
            .as_ref()?
            .wanted
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v.clone())
    }
    fn consumed(&self, slot: &Arc<Slot>) {
        // The actual renderer now owns the result. Its existing LRU controls residency.
        if let Some(movie) = self.state.lock().movie.as_mut() {
            movie.wanted.retain(|(_, s)| !Arc::ptr_eq(s, slot));
        }
        slot.canceled.store(true, Ordering::Release);
    }
    pub fn render(
        &self,
        d: &Device,
        family: Family,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
        recipe: Recipe,
    ) -> Result<RenderPipeline, RenderFailure> {
        let start = Instant::now();
        let slot = self.slot(recipe.key());
        let historical_ns = slot.as_ref().map_or(0, |s| s.creation_ns);
        let (p, warm) = if let Some(slot) = slot {
            let ready_before_lookup =
                slot.result.get().is_some() && slot.warm.load(Ordering::Acquire);
            let built = slot.result.get_or_try_init(|| {
                let p = d.create_render_direct(family, desc)?;
                Ok::<_, RenderFailure>(Compiled::Render(
                    p.raw().clone(),
                    p.lease().clone(),
                    p.0.metadata.get().and_then(|m| m.profile.get()).cloned(),
                ))
            })?;
            let Compiled::Render(raw, lease, profile) = built else {
                unreachable!()
            };
            let p = d.wrap(raw.clone(), lease.clone());
            if let Some(profile) = profile {
                let _ = p.meta().profile.set(profile.clone());
            }
            self.consumed(&slot);
            (p, ready_before_lookup)
        } else {
            (d.create_render_direct(family, desc)?, false)
        };
        p.meta().observation.get_or_init(|| {
            Observation::new(
                recipe,
                if warm {
                    historical_ns
                } else {
                    start.elapsed().as_nanos() as u64
                },
                warm.then(|| self.epoch.load(Ordering::Acquire)),
            )
        });
        Ok(p)
    }
    pub fn compute(
        &self,
        d: &Device,
        family: Family,
        desc: &wgpu::ComputePipelineDescriptor<'_>,
        recipe: Recipe,
    ) -> Result<ComputePipeline, RenderFailure> {
        let start = Instant::now();
        let slot = self.slot(recipe.key());
        let historical_ns = slot.as_ref().map_or(0, |s| s.creation_ns);
        let (p, warm) = if let Some(slot) = slot {
            let ready_before_lookup =
                slot.result.get().is_some() && slot.warm.load(Ordering::Acquire);
            let built = slot.result.get_or_try_init(|| {
                let p = d.create_compute_direct(family, desc)?;
                Ok::<_, RenderFailure>(Compiled::Compute(
                    p.raw().clone(),
                    p.lease().clone(),
                    p.0.metadata.get().and_then(|m| m.profile.get()).cloned(),
                ))
            })?;
            let Compiled::Compute(raw, lease, profile) = built else {
                unreachable!()
            };
            let p = d.wrap(raw.clone(), lease.clone());
            if let Some(profile) = profile {
                let _ = p.meta().profile.set(profile.clone());
            }
            self.consumed(&slot);
            (p, ready_before_lookup)
        } else {
            (d.create_compute_direct(family, desc)?, false)
        };
        p.meta().observation.get_or_init(|| {
            Observation::new(
                recipe,
                if warm {
                    historical_ns
                } else {
                    start.elapsed().as_nanos() as u64
                },
                warm.then(|| self.epoch.load(Ordering::Acquire)),
            )
        });
        Ok(p)
    }
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }
    pub fn mode(&self) -> &'static str {
        if self.consume { "consume" } else { "record" }
    }
    pub fn used(
        &self,
        obs: &Observation,
        epoch: u64,
        frame: Option<crate::pipeline_profile::Frame>,
        count: u64,
    ) {
        let mut state = self.state.lock();
        let Some(m) = state.movie.as_mut() else {
            return;
        };
        if self.epoch() != epoch || frame.is_some_and(|f| f.session != m.session) {
            return;
        }
        if m.used.len() >= MAX_ENTRIES && !m.used.contains_key(&obs.key) {
            return;
        }
        let e = m.used.entry(obs.key).or_insert_with(|| Entry {
            warm: obs.warm_epoch == Some(epoch),
            recipe: obs.recipe.clone(),
            uses: 0,
            first: frame.and_then(|f| f.packet_id.or(f.frame_id)).unwrap_or(0),
            ns: obs.ns,
        });
        if e.uses == 0 && obs.warm_epoch == Some(epoch) {
            m.warm_uses += 1;
        }
        e.uses = e.uses.saturating_add(count);
    }
    fn checksum(entries: &[Entry]) -> Option<String> {
        Some(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(entries).ok()?)
        ))
    }
    fn snapshot(m: &Movie) -> Manifest {
        Manifest {
            checksum: String::new(),
            schema: SCHEMA,
            identity: m.identity.clone(),
            entries: m.used.values().cloned().collect(),
            warm_uses: m.warm_uses,
        }
    }
    pub fn save(&self) {
        let m = self.state.lock().movie.as_ref().map(Self::snapshot);
        if let Some(m) = m {
            let _ = self.write(m);
        }
    }
    pub fn end(&self, session: Option<u64>) {
        let _transition = self.transition.lock();
        if let Some(last) = self.take_movie(session, true) {
            let _ = self.write(Self::snapshot(&last));
        }
    }
    fn take_movie(&self, session: Option<u64>, clear_programs: bool) -> Option<Movie> {
        let mut state = self.state.lock();
        if session.is_some_and(|id| state.movie.as_ref().is_none_or(|m| m.session != id)) {
            return None;
        }
        self.epoch.fetch_add(1, Ordering::AcqRel);
        if clear_programs {
            state.sources.agal.clear();
        }
        state.movie.take()
    }
    fn write(&self, mut m: Manifest) -> io::Result<()> {
        if m.entries.is_empty() {
            return Ok(());
        }
        m.checksum = Self::checksum(&m.entries).ok_or(io::ErrorKind::InvalidData)?;
        let bytes = serde_json::to_vec(&m)?;
        if bytes.len() as u64 > MAX_FILE {
            return Ok(());
        }
        let mut temp = tempfile::NamedTempFile::new_in(&self.dir)?;
        temp.write_all(&bytes)?;
        temp.persist(self.dir.join(format!("{}.recipes", m.identity)))?;
        let mut files: Vec<_> = fs::read_dir(&self.dir)?
            .flatten()
            .filter(|e| {
                e.file_type().is_ok_and(|t| t.is_file())
                    && e.path()
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .is_some_and(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()))
                    && e.path().extension().is_some_and(|s| s == "recipes")
            })
            .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
            .collect();
        files.sort();
        for (_, path) in files.iter().take(files.len().saturating_sub(MAX_FILES)) {
            let _ = fs::remove_file(path);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    #[ignore = "requires a Vulkan adapter and RUFFLE_PIPELINE_HISTORY=1"]
    fn foreground_pressure_session_cancellation_and_loss_release_prewarm_work() {
        let dir = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        let path = dir.path().join("cache");
        let device = super::super::tests::device(4 << 20);
        // SAFETY: only this test owns the private directory.
        unsafe {
            device.enable_pipeline_cache(&path, "test");
        }
        let h = device.history().expect("set RUFFLE_PIPELINE_HISTORY=1");
        let shader = device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0,0,0,1); }
                     @fragment fn fs() -> @location(0) vec4f { return vec4f(1); }"
                        .into(),
                ),
            })
            .unwrap();
        let create = || {
            device.create_render_pipeline_for(
                Family::Filter,
                &wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs"),
                        targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                        compilation_options: Default::default(),
                    }),
                    primitive: Default::default(),
                    multisample: Default::default(),
                    depth_stencil: None,
                    multiview_mask: None,
                    cache: None,
                },
            )
        };
        let used = |pipeline: RenderPipeline| {
            let mut uses = PipelineUses::default();
            pipeline.record_pipeline_use(&mut uses);
            uses.submitted(&device);
        };
        let barrier = || {
            let (tx, rx) = mpsc::channel();
            h.jobs
                .send(Box::new(move || {
                    tx.send(()).unwrap();
                }))
                .unwrap();
            rx.recv_timeout(Duration::from_secs(10)).unwrap();
        };
        let hold_worker = || {
            let (tx, rx) = mpsc::channel();
            let (ready, started) = mpsc::channel();
            h.jobs
                .send(Box::new(move || {
                    ready.send(()).unwrap();
                    rx.recv_timeout(Duration::from_secs(10)).unwrap();
                }))
                .unwrap();
            started.recv_timeout(Duration::from_secs(10)).unwrap();
            tx
        };
        h.begin(&device, 1, b"movie", "header", "High");
        used(create().unwrap());
        h.end(Some(1));
        let baseline = device.pipeline_compile_stats().0;
        let hold = hold_worker();
        h.begin(&device, 2, b"movie", "header", "High");
        assert_eq!(h.state.lock().movie.as_ref().unwrap().wanted.len(), 1);
        used(create().unwrap());
        h.end(Some(2));
        hold.send(()).unwrap();
        barrier();
        assert_eq!(device.pipeline_compile_stats().0, baseline + 1);

        let buffer = device
            .create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (4 << 20) - (16 << 10),
                usage: wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
            .unwrap();
        let retries = device.runtime().stats().creation_retries;
        h.begin(&device, 3, b"movie", "header", "High");
        barrier();
        assert!(!device.speculation_fits());
        assert_eq!(device.pipeline_compile_stats().0, baseline + 1);
        assert_eq!(device.runtime().stats().creation_retries, retries);
        drop(buffer);
        used(create().unwrap());
        h.end(Some(3));

        let hold = hold_worker();
        h.begin(&device, 4, b"movie", "header", "High");
        device.runtime().fail(RenderFailure::new(
            RenderFailureKind::DeviceLost,
            "test device lost",
        ));
        h.end(Some(4));
        hold.send(()).unwrap();
        barrier();
        assert_eq!(create().unwrap_err().kind(), RenderFailureKind::DeviceLost);
        let fresh = super::super::tests::device(4 << 20);
        assert_ne!(fresh.runtime().generation(), device.runtime().generation());
        assert!(fresh.runtime().check_health().is_ok());
    }
}

#[cfg(test)]
mod submission_tests;
