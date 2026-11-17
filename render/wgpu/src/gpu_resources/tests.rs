use super::*;
use crate::gpu_runtime::{GpuBudgetConfig, ResourceKind};

fn device(limit: u64) -> Device {
    let mut options = wgpu::InstanceDescriptor::new_without_display_handle();
    options.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(options);
    let (_, raw, queue) = futures::executor::block_on(crate::backend::request_adapter_and_device(
        wgpu::Backends::VULKAN,
        &instance,
        None,
        wgpu::PowerPreference::HighPerformance,
    ))
    .unwrap();
    let mut config = GpuBudgetConfig::for_adapter(wgpu::DeviceType::DiscreteGpu, false);
    config.hard_limit = limit;
    config.soft_limit = limit;
    let runtime = GpuRuntime::new(config).unwrap();
    runtime.install_device_callbacks(&raw);
    Device::new(raw, queue, runtime)
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn native_creation_failures_refund_reserved_storage() {
    let device = device(512);
    let too_large = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    assert_eq!(
        too_large.unwrap_err().kind(),
        RenderFailureKind::BudgetExceeded
    );
    assert_eq!(device.runtime().stats().total_bytes, 0);
    let invalid = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64,
        usage: wgpu::BufferUsages::empty(),
        mapped_at_creation: false,
    });
    assert_eq!(invalid.unwrap_err().kind(), RenderFailureKind::Validation);
    assert_eq!(device.runtime().stats().total_bytes, 0);
    assert_eq!(device.runtime().stats().resources, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn native_bindings_keep_storage_charged_without_double_counting_clones() {
    let device = device(4096);
    let buffer = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        })
        .unwrap();
    let alias = buffer.clone();
    let layout = device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
        .unwrap();
    let group = device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
        .unwrap();
    drop(buffer);
    drop(alias);
    assert_eq!(
        device.runtime().stats().kind_bytes[ResourceKind::Buffer as usize],
        256
    );
    drop(group);
    drop(layout);
    assert_eq!(device.runtime().stats().total_bytes, 0);
    assert_eq!(device.runtime().stats().resources, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn encoded_resources_survive_handle_drop_until_recording_is_discarded() {
    let device = device(4096);
    let mut encoder = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
        .unwrap();
    let buffer = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
        .unwrap();
    encoder.clear_buffer(&buffer, 0, None);
    drop(buffer);
    assert_eq!(device.runtime().stats().total_bytes, 256);
    let commands = encoder.finish().unwrap();
    assert_eq!(device.runtime().stats().total_bytes, 256);
    drop(commands);
    assert_eq!(device.runtime().stats().total_bytes, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn submitted_uploads_retire_only_after_gpu_completion() {
    let device = device(4096);
    let buffer = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
        .unwrap();
    let queue = device.queue();
    queue.write_buffer(&buffer, 0, &[7; 256]).unwrap();
    assert_eq!(
        device.runtime().stats().kind_bytes[ResourceKind::StagingBuffer as usize],
        256
    );
    drop(buffer);
    let submission = queue.submit([]).unwrap();
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    assert_eq!(device.runtime().stats().total_bytes, 0);
    assert_eq!(device.runtime().stats().resources, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn rejected_upload_does_not_write_or_keep_a_reservation() {
    let device = device(256);
    let buffer = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
        .unwrap();
    assert_eq!(
        device
            .queue()
            .write_buffer(&buffer, 0, &[7; 256])
            .unwrap_err()
            .kind(),
        RenderFailureKind::BudgetExceeded
    );
    assert_eq!(device.runtime().stats().reserved_bytes, 0);
    assert_eq!(device.runtime().stats().total_bytes, 256);
    drop(buffer);
    assert_eq!(device.runtime().stats().total_bytes, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn imported_texture_aliases_share_storage_and_reject_foreign_bindings() {
    let device = device(4096);
    let texture = device
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .unwrap();
    let imported = device
        .import_texture(texture.raw().clone(), ResourceKind::HostTarget)
        .unwrap();
    assert_eq!(texture.lease().id(), imported.lease().id());
    assert_eq!(device.runtime().stats().total_bytes, 1024);
    let foreign = device
        .raw()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[],
        });
    assert_eq!(
        device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &foreign,
                entries: &[]
            })
            .unwrap_err()
            .kind(),
        RenderFailureKind::Validation
    );
    drop(texture);
    drop(imported);
    assert_eq!(device.runtime().stats().total_bytes, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn renderer_initialization_obeys_the_shared_device_budget() {
    let mut options = wgpu::InstanceDescriptor::new_without_display_handle();
    options.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(options);
    let (adapter, raw, queue) =
        futures::executor::block_on(crate::backend::request_adapter_and_device(
            wgpu::Backends::VULKAN,
            &instance,
            None,
            wgpu::PowerPreference::HighPerformance,
        ))
        .unwrap();
    let mut config = GpuBudgetConfig::for_adapter(wgpu::DeviceType::DiscreteGpu, false);
    config.hard_limit = 1;
    config.soft_limit = 1;
    let runtime = GpuRuntime::new(config).unwrap();
    runtime.install_device_callbacks(&raw);
    let descriptors = crate::descriptors::Descriptors::with_runtime(
        instance,
        adapter,
        raw,
        queue,
        runtime.clone(),
    );
    assert_eq!(
        descriptors.unwrap_err().kind(),
        RenderFailureKind::BudgetExceeded
    );
    assert_eq!(runtime.stats().reserved_bytes, 0);
    assert_eq!(runtime.stats().total_bytes, 0);
    assert_eq!(runtime.stats().resources, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn staging_copy_is_charged_and_preserves_uploaded_bytes() {
    let device = device(8192);
    let buffer = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    let mut belt = StagingBelt::new(device.clone(), 64);
    belt.write_buffer(&mut encoder, &buffer, 0, wgpu::BufferSize::new(16).unwrap())
        .unwrap()
        .copy_from_slice(&[42; 16]);
    assert_eq!(
        device.runtime().stats().kind_bytes[ResourceKind::StagingBuffer as usize],
        16
    );
    belt.finish();
    let index = device.queue().submit([encoder.finish().unwrap()]).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    rx.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
    let bytes = buffer.get_mapped_range(..).unwrap();
    assert_eq!(&*bytes, &[42; 16]);
    drop(bytes);
    buffer.unmap();
    assert_eq!(
        device.runtime().stats().kind_bytes[ResourceKind::StagingBuffer as usize],
        0
    );
}

#[test]
#[ignore = "requires a Vulkan adapter with timestamp queries"]
fn profiler_queries_and_readback_storage_obey_the_device_ledger() {
    let device = device(16384);
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    let mut profiler = crate::gpu_profiler::GpuProfiler::new(
        &device,
        crate::gpu_profiler::GpuProfilerSettings {
            enable_timer_queries: true,
            ..Default::default()
        },
    )
    .unwrap();
    {
        let _scope = profiler.scope("measured frame", &mut encoder);
    }
    assert!(device.runtime().stats().kind_bytes[ResourceKind::QuerySet as usize] > 0);
    assert!(device.runtime().stats().kind_bytes[ResourceKind::ReadbackBuffer as usize] > 0);
    profiler.resolve_queries(&mut encoder);
    let submission = device.queue().submit([encoder.finish().unwrap()]).unwrap();
    profiler.end_frame().unwrap();
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    let timings = profiler
        .process_finished_frame(device.queue().get_timestamp_period())
        .unwrap();
    assert_eq!(timings[0].label, "measured frame");
    assert!(timings[0].time.is_some());
    drop(profiler);
    assert_eq!(device.runtime().stats().total_bytes, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter with timestamp queries"]
fn profiler_admission_failure_refunds_partial_pool_and_latches_failure() {
    let device = device(512);
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    let profiler = crate::gpu_profiler::GpuProfiler::new(
        &device,
        crate::gpu_profiler::GpuProfilerSettings {
            enable_timer_queries: true,
            ..Default::default()
        },
    )
    .unwrap();
    {
        let _scope = profiler.scope("cannot allocate a readback buffer", &mut encoder);
    }
    assert_eq!(
        device.runtime().failure().unwrap().kind(),
        RenderFailureKind::BudgetExceeded
    );
    assert_eq!(device.runtime().stats().reserved_bytes, 0);
    drop(profiler);
    drop(encoder);
    assert_eq!(device.runtime().stats().total_bytes, 0);
    assert_eq!(device.runtime().stats().resources, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn stage3d_buffer_failure_preserves_the_ledger_and_typed_cause() {
    use ruffle_render::backend::{BufferUsage, Context3D, Context3DProfile};
    let mut options = wgpu::InstanceDescriptor::new_without_display_handle();
    options.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(options);
    let (adapter, raw, queue) =
        futures::executor::block_on(crate::backend::request_adapter_and_device(
            wgpu::Backends::VULKAN,
            &instance,
            None,
            wgpu::PowerPreference::HighPerformance,
        ))
        .unwrap();
    let descriptors =
        Arc::new(crate::descriptors::Descriptors::new(instance, adapter, raw, queue).unwrap());
    let mut context =
        crate::context3d::WgpuContext3D::new(descriptors.clone(), Context3DProfile::Baseline)
            .unwrap();
    let before = descriptors.runtime.stats().total_bytes;
    let failure = match context.create_index_buffer(BufferUsage::StaticDraw, u32::MAX) {
        Err(failure) => failure,
        Ok(_) => panic!("oversize Stage3D buffer must be rejected"),
    };
    assert_eq!(failure.kind(), RenderFailureKind::BudgetExceeded);
    assert_eq!(descriptors.runtime.stats().reserved_bytes, 0);
    assert_eq!(descriptors.runtime.stats().total_bytes, before);
    context.health().fail(failure.clone());
    assert_eq!(context.health().check().unwrap_err(), failure);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn retryable_creation_reclaims_before_exactly_one_retry() {
    let device = device(4096);
    let mut attempts = 0;
    let result = device.create(ResourceKind::Buffer, 256, &[], |raw| {
        attempts += 1;
        if attempts == 1 {
            return Err(RenderFailure::new(
                RenderFailureKind::OutOfMemory,
                "injected creation failure",
            ));
        }
        Ok(raw.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    });
    assert!(
        result.is_ok(),
        "a clean candidate creation can retry: {result:?}"
    );
    assert_eq!(attempts, 2);
    assert_eq!(device.runtime().stats().reserved_bytes, 0);
    assert_eq!(device.runtime().stats().total_bytes, 256);
    drop(result);
    assert_eq!(device.runtime().stats().total_bytes, 0);
    let mut attempts = 0;
    let result: Result<Buffer, _> = device.create(ResourceKind::Buffer, 256, &[], |_| {
        attempts += 1;
        Err(RenderFailure::new(
            RenderFailureKind::OutOfMemory,
            "persistent creation failure",
        ))
    });
    assert_eq!(result.unwrap_err().kind(), RenderFailureKind::OutOfMemory);
    assert_eq!(attempts, 2);
    assert_eq!(device.runtime().stats().total_bytes, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn native_submissions_never_exceed_two_in_flight() {
    let device = device(4096);
    for _ in 0..16 {
        let commands = device
            .create_command_encoder(&Default::default())
            .unwrap()
            .finish()
            .unwrap();
        device.queue().submit([commands]).unwrap();
        let stats = device.runtime().stats();
        assert!(
            stats.submitted - stats.completed <= 2,
            "own queue depth must be bounded"
        );
        assert!(device.0.inner().pending_submissions.lock().unwrap().len() <= 2);
    }
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    assert_eq!(
        device.runtime().stats().submitted,
        device.runtime().stats().completed
    );
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn native_allocation_reclaims_idle_storage_before_exceeding_the_budget() {
    let device = device(1024);
    let budget = crate::idle_pool::Budget::with_runtime(device.runtime());
    let make = |size| {
        device
            .create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
            .unwrap()
    };
    let cached = make(512);
    cached.lease().set_idle(true);
    let pool = budget.key();
    budget.put(pool, 0, 512, cached);
    let live = make(512);
    assert_eq!(device.runtime().stats().idle_bytes, 512);
    let next = make(256);
    assert_eq!(device.runtime().stats().idle_bytes, 0);
    assert_eq!(device.runtime().stats().total_bytes, 768);
    assert!(budget.take(pool, |_: &Buffer| Some(())).is_none());
    drop((live, next));
    assert_eq!(device.runtime().stats().total_bytes, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn encoder_retains_only_used_resources_with_interleaved_recordings() {
    let device = device(4096);
    let buffer = |size| {
        device
            .create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
            .unwrap()
    };
    let unused = buffer(1024);
    let a = buffer(128);
    let b = buffer(256);
    let mut first = device.create_command_encoder(&Default::default()).unwrap();
    let mut second = device.create_command_encoder(&Default::default()).unwrap();
    first.clear_buffer(&a, 0, None);
    second.clear_buffer(&b, 0, None);
    drop(unused);
    drop(a);
    drop(b);
    assert_eq!(
        device.runtime().stats().total_bytes,
        384,
        "unrelated storage must not be retained by an encoder"
    );
    let first = first.finish().unwrap();
    let second = second.finish().unwrap();
    drop(first);
    assert_eq!(
        device.runtime().stats().total_bytes,
        256,
        "the second recording must not retain the first one's uses"
    );
    drop(second);
    assert_eq!(device.runtime().stats().total_bytes, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn temporary_material_cache_respects_ranges_layouts_lru_and_reclamation() {
    let device = device(8192);
    let cache = crate::material_cache::MaterialCache::new(device.runtime(), 2);
    let buffer = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        })
        .unwrap();
    let layout_desc = wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    };
    let first_layout = device.create_bind_group_layout(&layout_desc).unwrap();
    let second_layout = device.create_bind_group_layout(&layout_desc).unwrap();
    let get = |layout: &BindGroupLayout, offset| {
        cache
            .get_or_create(
                &device,
                &wgpu::BindGroupDescriptor {
                    label: None,
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buffer,
                            offset,
                            size: wgpu::BufferSize::new(256),
                        }),
                    }],
                },
            )
            .unwrap()
    };
    let first = get(&first_layout, 0);
    assert_eq!(first, get(&first_layout, 0));
    let range = get(&first_layout, 256);
    assert_ne!(first, range);
    assert_eq!(first, get(&first_layout, 0)); // first is MRU
    let layout = get(&second_layout, 0);
    assert_ne!(first, layout);
    assert_ne!(range, get(&first_layout, 256)); // range was LRU
    assert!(device.runtime().stats().cache_retained_bytes > 0);
    device.runtime().scheduler().reclaim();
    assert_eq!(device.runtime().stats().cache_retained_bytes, 0);
    assert_ne!(first, get(&first_layout, 0));
    // Recorded handles remain usable after eviction; storage still belongs to them.
    drop(cache);
    assert_eq!(device.runtime().stats().total_bytes, 1024);
    drop((first, range, layout, buffer));
    assert_eq!(device.runtime().stats().total_bytes, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn render_pass_suppresses_identical_state_but_preserves_offsets_and_boundaries() {
    let device = device(8192);
    let buffer = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        })
        .unwrap();
    let layout = device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
        .unwrap();
    let group = device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(256),
                }),
            }],
        })
        .unwrap();
    let texture = device
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .unwrap();
    let view = texture.create_view(&Default::default()).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    for _ in 0..2 {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        for _ in 0..3 {
            pass.set_bind_group(0, group.raw(), &[0]);
            pass.set_vertex_buffer(0, buffer.slice(0..64));
            pass.set_index_buffer(buffer.slice(0..64), wgpu::IndexFormat::Uint16);
            pass.set_scissor_rect(0, 0, 4, 4);
            pass.set_stencil_reference(1);
        }
        assert_eq!(pass.state_changes().bind_groups, 1);
        assert_eq!(pass.state_changes().vertex_buffers, 1);
        assert_eq!(pass.state_changes().index_buffers, 1);
        assert_eq!(pass.state_changes().scissors, 1);
        assert_eq!(pass.state_changes().stencil_references, 1);
        pass.set_bind_group(0, group.raw(), &[256]);
        pass.set_vertex_buffer(0, buffer.slice(64..128));
        pass.set_index_buffer(buffer.slice(0..64), wgpu::IndexFormat::Uint32);
        pass.set_scissor_rect(4, 0, 4, 4);
        pass.set_stencil_reference(2);
        assert_eq!(pass.state_changes().bind_groups, 2);
        assert_eq!(pass.state_changes().vertex_buffers, 2);
        assert_eq!(pass.state_changes().index_buffers, 2);
        assert_eq!(pass.state_changes().scissors, 2);
        assert_eq!(pass.state_changes().stencil_references, 2);
    }
    assert!(encoder.finish().is_ok());
}
