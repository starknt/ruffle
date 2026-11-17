use crate::buffer_builder::BufferBuilder;
use crate::filters::{FilterVertex, Filters};
use crate::layouts::BindLayouts;
use crate::shaders::Shaders;
use crate::{BitmapSamplers, Pipelines, PosColorVertex, PosUvVertex, PosVertex};
use fnv::FnvHashMap;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use wgpu::Backend;

pub struct Descriptors {
    pub runtime: crate::gpu_runtime::GpuRuntime,
    pub wgpu_instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: crate::gpu_resources::Device,
    pub limits: wgpu::Limits,
    pub backend: Backend,
    pub queue: crate::gpu_resources::Queue,
    pub submission_lock: Option<Arc<Mutex<()>>>,
    pub bitmap_samplers: BitmapSamplers,
    pub bind_layouts: BindLayouts,
    pub quad: Quad,
    copy_pipeline:
        Mutex<FnvHashMap<(u32, wgpu::TextureFormat), crate::gpu_resources::RenderPipeline>>,
    pub shaders: Shaders,
    pipelines: Mutex<FnvHashMap<(u32, wgpu::TextureFormat), Arc<Pipelines>>>,
    pub filters: Filters,
}

impl Debug for Descriptors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Descriptors")
    }
}

impl Descriptors {
    pub fn new(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let runtime =
            crate::gpu_runtime::GpuRuntime::new(crate::gpu_runtime::GpuBudgetConfig::for_adapter(
                adapter.get_info().device_type,
                cfg!(target_os = "android"),
            ))?;
        runtime.install_device_callbacks(&device);
        Self::with_runtime(instance, adapter, device, queue, runtime)
    }

    /// Shared-device hosts install callbacks once, then use this constructor.
    pub fn with_runtime(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        runtime: crate::gpu_runtime::GpuRuntime,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let device = crate::gpu_resources::Device::new(device, queue, runtime);
        Self::with_device(instance, adapter, device)
    }

    /// Reuse the native factory and weak resource index across shared host targets.
    pub fn with_device(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: crate::gpu_resources::Device,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let runtime = device.runtime().clone();
        runtime.check_health()?;
        let device =
            device.with_owner(runtime.issue_owner(crate::gpu_runtime::OwnerKind::Renderer)?);
        let queue = device.queue();
        let limits = device.limits();
        let bind_layouts = BindLayouts::new(&device)?;
        let bitmap_samplers = BitmapSamplers::new(&device)?;
        let shaders = Shaders::new(&device)?;
        let quad = Quad::new(&device)?;
        let filters = Filters::new(&device)?;
        let backend = adapter.get_info().backend;

        runtime.check_health()?;
        Ok(Self {
            runtime,
            wgpu_instance: instance,
            adapter,
            device,
            limits,
            backend,
            queue,
            submission_lock: None,
            bitmap_samplers,
            bind_layouts,
            quad,
            copy_pipeline: Default::default(),
            shaders,
            pipelines: Default::default(),
            filters,
        })
    }

    pub fn lock_submission(&self) -> Option<std::sync::MutexGuard<'_, ()>> {
        self.submission_lock.as_ref().map(|lock| {
            lock.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        })
    }

    pub fn copy_pipeline(
        &self,
        format: wgpu::TextureFormat,
        msaa_sample_count: u32,
    ) -> Result<crate::gpu_resources::RenderPipeline, crate::gpu_runtime::RenderFailure> {
        let mut pipelines = self
            .copy_pipeline
            .lock()
            .expect("Pipelines should not be already locked");
        Ok(
            crate::gpu_resources::try_cached(&mut pipelines, (msaa_sample_count, format), || {
                let copy_texture_pipeline_layout =
                    &self
                        .device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: create_debug_label!("Copy pipeline layout").as_deref(),
                            bind_group_layouts: &[None, None, Some(&self.bind_layouts.bitmap)],
                            immediate_size: 0,
                        })?;
                self.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: create_debug_label!("Copy pipeline").as_deref(),
                        layout: Some(copy_texture_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &self.shaders.copy_shader,
                            entry_point: Some("main_vertex"),
                            buffers: &[],
                            compilation_options: Default::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &self.shaders.copy_shader,
                            entry_point: Some("main_fragment"),
                            targets: &[Some(wgpu::ColorTargetState {
                                format,
                                // All of our blending has been done by now, so we want
                                // to overwrite the target pixels without any blending
                                blend: Some(wgpu::BlendState::REPLACE),
                                write_mask: Default::default(),
                            })],
                            compilation_options: Default::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            strip_index_format: None,
                            front_face: wgpu::FrontFace::Ccw,
                            cull_mode: None,
                            polygon_mode: wgpu::PolygonMode::default(),
                            unclipped_depth: false,
                            conservative: false,
                        },
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState {
                            count: msaa_sample_count,
                            mask: !0,
                            alpha_to_coverage_enabled: false,
                        },
                        multiview_mask: None,
                        cache: None,
                    })
            })?
            .clone(),
        )
    }

    pub fn pipelines(
        &self,
        msaa_sample_count: u32,
        format: wgpu::TextureFormat,
    ) -> Result<Arc<Pipelines>, crate::gpu_runtime::RenderFailure> {
        let mut pipelines = self
            .pipelines
            .lock()
            .expect("Pipelines should not be already locked");
        Ok(
            crate::gpu_resources::try_cached(&mut pipelines, (msaa_sample_count, format), || {
                Ok::<_, crate::gpu_runtime::RenderFailure>(Arc::new(Pipelines::new(
                    &self.device,
                    &self.shaders,
                    format,
                    msaa_sample_count,
                    &self.bind_layouts,
                )?))
            })?
            .clone(),
        )
    }
}

pub struct Quad {
    pub vertex_buffer: crate::gpu_resources::Buffer,
    pub vertices_pos: Range<wgpu::BufferAddress>,
    pub vertices_pos_uv: Range<wgpu::BufferAddress>,
    pub vertices_pos_color: Range<wgpu::BufferAddress>,
    pub vertices_filter: Range<wgpu::BufferAddress>,
    pub index_buffer: crate::gpu_resources::Buffer,
    pub indices_quad: Range<u32>,
    pub indices_line: Range<u32>,
    pub indices_line_rect: Range<u32>,
    pub full_index_range: Range<wgpu::BufferAddress>,
}

impl Quad {
    pub fn new(
        device: &crate::gpu_resources::Device,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let mut vertex_buffer = BufferBuilder::new(&device.limits(), 0);
        let vertices_pos = vertex_buffer
            .add(&[
                PosVertex {
                    position: [0.0, 0.0],
                },
                PosVertex {
                    position: [1.0, 0.0],
                },
                PosVertex {
                    position: [1.0, 1.0],
                },
                PosVertex {
                    position: [0.0, 1.0],
                },
            ])
            .expect("Quad vertices must fit");
        let vertices_pos_uv = vertex_buffer
            .add(&[
                PosUvVertex {
                    position: [0.0, 0.0],
                    uv: [0.0, 0.0, 1.0],
                },
                PosUvVertex {
                    position: [1.0, 0.0],
                    uv: [1.0, 0.0, 1.0],
                },
                PosUvVertex {
                    position: [1.0, 1.0],
                    uv: [1.0, 1.0, 1.0],
                },
                PosUvVertex {
                    position: [0.0, 1.0],
                    uv: [0.0, 1.0, 1.0],
                },
            ])
            .expect("Quad vertices must fit");
        let vertices_pos_color = vertex_buffer
            .add(&[
                PosColorVertex {
                    position: [0.0, 0.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                },
                PosColorVertex {
                    position: [1.0, 0.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                },
                PosColorVertex {
                    position: [1.0, 1.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                },
                PosColorVertex {
                    position: [0.0, 1.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                },
            ])
            .expect("Quad vertices must fit");
        let filter_vertices = vertex_buffer
            .add(&[
                FilterVertex {
                    position: [0.0, 0.0],
                    uv: [0.0, 0.0],
                },
                FilterVertex {
                    position: [1.0, 0.0],
                    uv: [1.0, 0.0],
                },
                FilterVertex {
                    position: [1.0, 1.0],
                    uv: [1.0, 1.0],
                },
                FilterVertex {
                    position: [0.0, 1.0],
                    uv: [0.0, 1.0],
                },
            ])
            .expect("Filter vertices must fit");

        let vbo = vertex_buffer.finish(
            device,
            Some("Quad vertices".to_string()),
            wgpu::BufferUsages::VERTEX,
        )?;

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad indices"),
            contents: bytemuck::cast_slice(&[0, 1, 2, 0, 2, 3, 0, 1, 2, 3, 0]),
            usage: wgpu::BufferUsages::INDEX,
        })?;
        let indices_quad = 0..6;
        let indices_line = 0..2;
        let indices_line_rect = 6..11;
        let full_index_range = 0..index_buffer.size();

        Ok(Self {
            vertex_buffer: vbo,
            vertices_pos,
            vertices_pos_uv,
            vertices_pos_color,
            vertices_filter: filter_vertices,
            index_buffer,
            indices_quad,
            indices_line,
            indices_line_rect,
            full_index_range,
        })
    }
}
