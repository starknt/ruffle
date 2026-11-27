use super::*;
use crate::gpu_resources::tests::device;

#[derive(Clone, Copy)]
enum Draw {
    Direct,
    Indirect,
    Multi,
    Count,
}

fn observe_submissions(indirect: bool) {
    let root = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
    let d = device(8 << 20);
    // SAFETY: this test owns the cache directory.
    unsafe { d.enable_pipeline_cache(&root.path().join("gpu"), "submit-use-test") };
    d.enable_pipeline_profile();
    let history = d.history();
    if let Some(h) = history {
        h.begin(&d, 0, b"first", "header", "High");
    }
    let shader = d
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0,0,0,1); }
             @fragment fn fs() -> @location(0) vec4f { return vec4f(1); }"
                    .into(),
            ),
        })
        .unwrap();
    let pipeline = d
        .create_render_pipeline_for(
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
        .unwrap();
    let view = d
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d::default(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .unwrap()
        .create_view(&Default::default())
        .unwrap();
    let indices = d
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0; 8],
            usage: wgpu::BufferUsages::INDEX,
        })
        .unwrap();
    let args = d
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[3u32, 1, 0, 0, 0, 1]),
            usage: wgpu::BufferUsages::INDIRECT,
        })
        .unwrap();
    let record = |draw, id| {
        let _frame = d.pipeline_frame(Some(id), Some(id + 100));
        let mut encoder = d.create_command_encoder(&Default::default()).unwrap();
        let targets = [Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations::default(),
        })];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &targets,
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
        match draw {
            Draw::Direct => pass.draw_indexed(0..3, 0, 0..1),
            Draw::Indirect => pass.draw_indexed_indirect(&args, 0),
            Draw::Multi => pass.multi_draw_indexed_indirect(&args, 0, 1),
            Draw::Count => pass.multi_draw_indexed_indirect_count(&args, 0, 20, 1),
        }
        drop(pass);
        encoder.finish().unwrap()
    };
    let counts = || {
        let profile = d
            .pipeline_profile()
            .unwrap()
            .snapshot(d.runtime().generation(), Instant::now());
        profile["variants"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["kind"] == "render")
            .unwrap()
            .clone()
    };
    if indirect {
        let mut draws = vec![Draw::Indirect, Draw::Multi];
        if d.features()
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
        {
            draws.push(Draw::Count);
        }
        for (n, draw) in draws.into_iter().enumerate() {
            d.queue().submit([record(draw, n as u64 + 1)]).unwrap();
            assert_eq!(counts()["uses"], n + 1);
            if let Some(h) = history {
                assert_eq!(
                    h.state
                        .lock()
                        .movie
                        .as_ref()
                        .unwrap()
                        .used
                        .values()
                        .next()
                        .unwrap()
                        .uses,
                    n as u64 + 1
                );
            }
        }
        return;
    }
    drop(record(Draw::Direct, 1));
    assert_eq!(counts()["uses"], 0, "abandoned command buffer");
    if let Some(h) = history {
        assert!(h.state.lock().movie.as_ref().unwrap().used.is_empty());
    }
    let accepted = record(Draw::Direct, 2);
    std::thread::scope(|scope| scope.spawn(|| d.queue().submit([accepted])).join().unwrap())
        .unwrap();
    assert_eq!(counts()["uses"], 1);
    assert_eq!(counts()["firstUse"]["frameId"], 2);
    let old = record(Draw::Direct, 3);
    if let Some(h) = history {
        h.quality(&d, 0, "Low".into());
    }
    d.queue().submit([old]).unwrap();
    if let Some(h) = history {
        assert!(
            h.state.lock().movie.as_ref().unwrap().used.is_empty(),
            "previous quality polluted the new manifest"
        );
    }
    let failed = record(Draw::Direct, 4);
    d.runtime().fail(RenderFailure::new(
        RenderFailureKind::DeviceLost,
        "unit test",
    ));
    assert!(d.queue().submit([failed]).is_err());
    assert_eq!(counts()["uses"], 2);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn only_successful_submissions_record_their_original_frame_and_movie() {
    observe_submissions(false);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn all_indirect_draw_commands_record_the_submitted_pipeline() {
    observe_submissions(true);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn abandoned_compute_dispatches_do_not_count_toward_submitted_uses() {
    use crate::capabilities::{NativeGpuUse, device_descriptor};
    use crate::gpu_runtime::GpuBudgetConfig;
    use futures::executor::block_on;

    let mut options = wgpu::InstanceDescriptor::new_without_display_handle();
    options.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(options);
    let adapter = block_on(instance.request_adapter(&Default::default())).unwrap();
    let (raw, queue) =
        block_on(adapter.request_device(&device_descriptor(&adapter, NativeGpuUse::SharedStage)))
            .unwrap();
    let runtime = GpuRuntime::new(GpuBudgetConfig::for_adapter(
        adapter.get_info().device_type,
        false,
    ))
    .unwrap();
    runtime.install_device_callbacks(&raw);
    let d = Device::new(raw, queue, runtime);
    d.enable_pipeline_profile();
    let shader = d
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage,read_write> words: array<u32>;
             @compute @workgroup_size(1) fn main() { words[0] = 1u; }"
                    .into(),
            ),
        })
        .unwrap();
    let pipeline = d
        .create_compute_pipeline_for(
            Family::Filter,
            &wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        )
        .unwrap();
    let buffer = d
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
        .unwrap();
    let group = d
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0).unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
        .unwrap();
    let record = || {
        let mut encoder = d.create_command_encoder(&Default::default()).unwrap();
        encoder
            .begin_compute_pass(&Default::default())
            .dispatch(&pipeline, &group, [1, 1]);
        encoder.finish().unwrap()
    };
    let uses = || {
        let snapshot = d
            .pipeline_profile()
            .unwrap()
            .snapshot(d.runtime().generation(), Instant::now());
        snapshot["variants"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["kind"] == "compute")
            .unwrap()["uses"]
            .as_u64()
            .unwrap()
    };
    drop(record());
    assert_eq!(uses(), 0);
    d.queue().submit([record(), record()]).unwrap();
    assert_eq!(uses(), 2);
}
