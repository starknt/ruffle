use super::*;
use crate::gpu_runtime::{GpuBudgetConfig, ResourceKind};

fn device(limit: u64) -> Device {
    device_with_staging_budget(limit, 64 * 1024 * 1024)
}

fn device_with_staging_budget(limit: u64, staging: u64) -> Device {
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
    config.staging_limit = staging;
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
fn repeated_recorded_raw_resources_remain_resolvable_until_encoder_finish() {
    let device = device(4096);
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    let buffer = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
        .unwrap();
    let raw = buffer.raw().clone();
    encoder.clear_buffer(&buffer, 0, Some(128));
    drop(buffer);
    // The recording already owns this resource; repeating a use cannot lose
    // its identity just because the caller released its original wrapper.
    encoder.clear_buffer(&raw, 128, Some(128));
    device.runtime().check_health().unwrap();
    let commands = encoder.finish().unwrap();
    drop(raw);
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
        64
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
        64,
        "completed page is available for reuse"
    );
    device.runtime().scheduler().reclaim();
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
    profiler.end_frame(&encoder).unwrap();
    device.poll(wgpu::PollType::Poll).unwrap();
    assert!(
        profiler
            .process_finished_frame(device.queue().get_timestamp_period())
            .is_none(),
        "a stage whose encoder is still on the worker queue has no GPU timings"
    );
    let submission = device.queue().submit([encoder.finish().unwrap()]).unwrap();
    device.runtime().purge_idle_and_cached_resources();
    assert!(
        device.runtime().stats().kind_bytes[ResourceKind::QuerySet as usize] > 0,
        "a pending profiler frame must survive idle reclamation"
    );
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
    assert!(
        device.runtime().stats().idle_bytes > 0,
        "completed query pools must participate in shared idle reclamation"
    );
    device.runtime().purge_idle_and_cached_resources();
    assert_eq!(
        device.runtime().stats().total_bytes,
        0,
        "idle query storage must be reclaimable while the profiler is alive"
    );

    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    {
        let _scope = profiler.scope("measured after reclamation", &mut encoder);
    }
    profiler.resolve_queries(&mut encoder);
    profiler.end_frame(&encoder).unwrap();
    let submission = device.queue().submit([encoder.finish().unwrap()]).unwrap();
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    let timings = profiler
        .process_finished_frame(device.queue().get_timestamp_period())
        .unwrap();
    assert_eq!(timings[0].label, "measured after reclamation");
    assert!(timings[0].time.is_some());
    assert!(device.runtime().stats().idle_bytes > 0);
    profiler
        .change_settings(crate::gpu_profiler::GpuProfilerSettings {
            enable_timer_queries: false,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        device.runtime().stats().total_bytes,
        0,
        "disabling profiling must release its actual idle query cache"
    );
    drop(profiler);
    device
        .runtime()
        .check_health()
        .expect("retiring recycled profiler buffers is valid");
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
fn configure_retries_with_the_queue_unlocked_during_reclamation() {
    use crate::resource_scheduler::{Reclaim, ReclaimClass};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Probe(Arc<Mutex<()>>, AtomicBool);
    impl Reclaim for Probe {
        fn oldest(&self) -> Option<u64> {
            self.1.store(self.0.try_lock().is_ok(), Ordering::Relaxed);
            None
        }
        fn evict_oldest(&self) -> bool {
            unreachable!()
        }
    }

    let device = device(4096);
    let gate = Arc::new(Mutex::new(()));
    let probe = Arc::new(Probe(gate.clone(), AtomicBool::new(false)));
    device
        .runtime()
        .scheduler()
        .register(ReclaimClass::Idle, probe.clone());
    let mut attempts = 0;
    device
        .configure_with(Some(&gate), |_| {
            assert!(
                gate.try_lock().is_err(),
                "raw configuration must exclude submissions"
            );
            attempts += 1;
            if attempts == 1 {
                return Err(RenderFailure::new(
                    RenderFailureKind::OutOfMemory,
                    "injected surface failure",
                ));
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(attempts, 2);
    assert!(
        probe.1.load(Ordering::Relaxed),
        "reclamation must permit queue progress"
    );
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
    let metrics = ruffle_render::profile::RenderMetrics::default();
    let _scope = ruffle_render::profile::enter(Some(&metrics));
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
    assert_eq!(metrics.snapshot().work.native_bind_groups, 4);
    assert_eq!(metrics.snapshot().work.native_vertex_buffers, 4);
    assert_eq!(metrics.snapshot().work.native_index_buffers, 4);
    assert_eq!(metrics.snapshot().work.native_scissors, 4);
    assert_eq!(metrics.snapshot().work.native_stencil_references, 4);
    assert!(encoder.finish().is_ok());
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn upload_arena_pages_pack_slices_and_preserve_unsubmitted_encoders() {
    let device = device(8192);
    let target = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
        .unwrap();
    let mut arena = crate::frame_upload::FrameUploadArena::new(device.clone(), 64);
    let mut first = device.create_command_encoder(&Default::default()).unwrap();
    for (offset, value) in [(0, 11), (16, 22)] {
        arena
            .write_buffer(
                &mut first,
                &target,
                offset,
                wgpu::BufferSize::new(16).unwrap(),
            )
            .unwrap()
            .copy_from_slice(&[value; 16]);
    }
    assert_eq!(
        device.runtime().stats().kind_counts[ResourceKind::StagingBuffer as usize],
        1
    );
    assert_eq!(
        device.runtime().stats().kind_bytes[ResourceKind::StagingBuffer as usize],
        64
    );
    arena.finish();
    let mut second = device.create_command_encoder(&Default::default()).unwrap();
    arena
        .write_buffer(&mut second, &target, 32, wgpu::BufferSize::new(16).unwrap())
        .unwrap()
        .copy_from_slice(&[33; 16]);
    arena.finish();
    assert_eq!(
        device.runtime().stats().kind_counts[ResourceKind::StagingBuffer as usize],
        2,
        "unsent pages cannot be recycled"
    );
    let index = device
        .queue()
        .submit([first.finish().unwrap(), second.finish().unwrap()])
        .unwrap();
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    arena.recall();
    for value in 44..49u8 {
        let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
        for part in 0..4u8 {
            arena
                .write_buffer(
                    &mut encoder,
                    &target,
                    u64::from(part) * 16,
                    wgpu::BufferSize::new(16).unwrap(),
                )
                .unwrap()
                .copy_from_slice(&[value + part; 16]);
        }
        arena.finish();
        let index = device.queue().submit([encoder.finish().unwrap()]).unwrap();
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: Some(Duration::from_secs(2)),
            })
            .unwrap();
        arena.recall();
    }
    assert_eq!(
        device.runtime().stats().kind_counts[ResourceKind::StagingBuffer as usize],
        2,
        "completed slots must reuse pages after wraparound"
    );
    let (tx, rx) = std::sync::mpsc::channel();
    target.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = target.get_mapped_range(..).unwrap();
    for (range, value) in [(0..16, 48), (16..32, 49), (32..48, 50), (48..64, 51)] {
        assert!(mapped[range].iter().all(|&byte| byte == value));
    }
    drop(mapped);
    target.unmap();
    device.runtime().scheduler().reclaim();
    assert_eq!(
        device.runtime().stats().kind_bytes[ResourceKind::StagingBuffer as usize],
        0
    );
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn large_texture_upload_slices_padded_layers_in_the_same_operation() {
    let device = device_with_staging_budget(8192, 512);
    let size = wgpu::Extent3d {
        width: 4,
        height: 5,
        depth_or_array_layers: 2,
    };
    let texture = device
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
        .unwrap();
    let mut data = vec![255; 16 + 2 * 7 * 32];
    for z in 0..2 {
        for y in 0..5 {
            let start = 16 + z * 7 * 32 + y * 32;
            data[start..start + 16].fill((z * 10 + y) as u8);
        }
    }
    device
        .queue()
        .write_texture(
            texture.as_image_copy(),
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 16,
                bytes_per_row: Some(32),
                rows_per_image: Some(7),
            },
            size,
        )
        .unwrap();
    assert!(
        device.runtime().stats().submitted > 0,
        "mandatory upload must submit bounded slices"
    );
    assert!(device.runtime().stats().kind_bytes[ResourceKind::StagingBuffer as usize] <= 512);
    let output = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 2560,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(5),
            },
        },
        size,
    );
    let index = device.queue().submit([encoder.finish().unwrap()]).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    output.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = output.get_mapped_range(..).unwrap();
    for z in 0..2 {
        for y in 0..5 {
            let start = (z * 5 + y) * 256;
            assert!(
                mapped[start..start + 16]
                    .iter()
                    .all(|&value| value == (z * 10 + y) as u8)
            );
        }
    }
    drop(mapped);
    output.unmap();
}

#[test]
#[ignore = "requires a Vulkan adapter with BC textures"]
fn compressed_mip_uploads_keep_physical_tail_blocks() {
    let device = device_with_staging_budget(8192, 512);
    assert!(
        device
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
    );
    let desc = wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 2,
        },
        mip_level_count: 4,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bc1RgbaUnorm,
        usage: wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    let data: Vec<u8> = (0..112).collect();
    let texture = device
        .create_texture_with_data(
            &device.queue(),
            &desc,
            wgpu::util::TextureDataOrder::LayerMajor,
            &data,
        )
        .unwrap();
    let output = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 3,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
    );
    let index = device.queue().submit([encoder.finish().unwrap()]).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    output.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = output.get_mapped_range(..).unwrap();
    assert_eq!(&mapped[..8], &data[104..112]);
    drop(mapped);
    output.unmap();
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn dynamic_upload_pages_preserve_alignment_and_retire_by_submission() {
    let device = device(8192);
    let mut arena = crate::frame_upload::FrameUploadArena::new(device.clone(), 1024);
    let first = arena.allocate_dynamic(100, 256).unwrap();
    let second = arena.allocate_dynamic(100, 256).unwrap();
    assert_eq!(first.buffer, second.buffer);
    assert_eq!(first.offset, 0);
    assert_eq!(second.offset, 256);
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    encoder.clear_buffer(&first.buffer, 0, None);
    let index = device.queue().submit([encoder.finish().unwrap()]).unwrap();
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    let reused = arena.allocate_dynamic(100, 256).unwrap();
    assert_eq!(reused.buffer, first.buffer);
    assert_eq!(reused.offset, 0);
    assert_eq!(
        device.runtime().stats().kind_bytes[ResourceKind::DynamicBuffer as usize],
        1024
    );
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn small_uniform_batches_share_a_page_without_shortening_bindings() {
    let device = device(8192);
    let mut arena = crate::frame_upload::FrameUploadArena::new(device.clone(), 1024);
    let mut allocations = Vec::new();
    for _ in 0..4 {
        allocations.push(arena.allocate_uniform(16, 768, 64).unwrap());
    }
    for (index, allocation) in allocations.iter().enumerate() {
        assert_eq!(allocation.buffer, allocations[0].buffer);
        assert_eq!(allocation.offset, index as u64 * 64);
        assert!(allocation.offset + 768 <= allocation.buffer.size());
    }
    assert_eq!(
        device.runtime().stats().kind_bytes[ResourceKind::DynamicBuffer as usize],
        1024
    );
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn large_staged_geometry_stream_splits_without_replaying_earlier_writes() {
    let device = device_with_staging_budget(16384, 512);
    let target = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4112,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    encoder.push_debug_group("spans bounded submissions");
    let mut shader_belt = StagingBelt::new(device.clone(), 64);
    shader_belt
        .write_buffer(&mut encoder, &target, 0, wgpu::BufferSize::new(16).unwrap())
        .unwrap()
        .copy_from_slice(&[7; 16]);
    let mut belt = StagingBelt::new(device.clone(), 128);
    let mut builder = crate::buffer_builder::BufferBuilder::new(&device.limits(), 0);
    let data: Vec<u8> = (0..4096).map(|v| (v % 251) as u8).collect();
    builder.add(&data).unwrap();
    builder
        .copy_to_at(&mut belt, &mut encoder, &target, 16)
        .unwrap();
    assert!(device.runtime().stats().submitted > 0);
    encoder.pop_debug_group();
    belt.finish();
    shader_belt.finish();
    let index = device.queue().submit([encoder.finish().unwrap()]).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    target.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = target.get_mapped_range(..).unwrap();
    assert_eq!(&mapped[..16], &[7; 16]);
    assert_eq!(&mapped[16..4112], &data);
    drop(mapped);
    target.unmap();
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn upload_page_reuse_waits_for_every_unsubmitted_encoder() {
    let device = device(8192);
    let target = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
        .unwrap();
    let mut arena = StagingBelt::new(device.clone(), 64);
    let mut first = device.create_command_encoder(&Default::default()).unwrap();
    let mut second = device.create_command_encoder(&Default::default()).unwrap();
    arena
        .write_buffer(&mut first, &target, 0, wgpu::BufferSize::new(16).unwrap())
        .unwrap()
        .copy_from_slice(&[11; 16]);
    arena
        .write_buffer(&mut second, &target, 16, wgpu::BufferSize::new(16).unwrap())
        .unwrap()
        .copy_from_slice(&[22; 16]);
    arena.finish();
    let index = device.queue().submit([first.finish().unwrap()]).unwrap();
    arena.recall();
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    let mut third = device.create_command_encoder(&Default::default()).unwrap();
    arena
        .write_buffer(&mut third, &target, 32, wgpu::BufferSize::new(32).unwrap())
        .unwrap()
        .copy_from_slice(&[33; 32]);
    arena.finish();
    assert_eq!(
        device.runtime().stats().kind_counts[ResourceKind::StagingBuffer as usize],
        2
    );
    let index = device
        .queue()
        .submit([second.finish().unwrap(), third.finish().unwrap()])
        .unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    target.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = target.get_mapped_range(..).unwrap();
    assert_eq!(&mapped[16..32], &[22; 16]);
    drop(mapped);
    target.unmap();
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn queued_stage_copy_precedes_a_later_upload_after_a_successor_is_cancelled() {
    let device = device(2 * 1024 * 1024);
    let queue = device.queue();
    let source = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
        .unwrap();
    let saved = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
        .unwrap();
    queue.write_buffer(&source, 0, &[12; 4]).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    encoder.copy_buffer_to_buffer(&source, 0, &saved, 0, 4);
    let turn = queue.reserve_submission().unwrap();
    let successor = queue.reserve_submission().unwrap();
    drop(successor);
    let (started, ready) = crossbeam_channel::bounded(1);
    let (wrote, observed) = crossbeam_channel::bounded(1);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            started.send(()).unwrap();
            queue.write_buffer(&source, 0, &[13; 4]).unwrap();
            wrote.send(()).unwrap();
            queue.submit([]).unwrap();
        });
        ready.recv_timeout(Duration::from_secs(2)).unwrap();
        let early = observed.recv_timeout(Duration::from_millis(40)).is_ok();
        turn.submit([encoder.finish().unwrap()]).unwrap();
        assert!(!early, "a reliable upload overtook an accepted stage");
    });
    let (tx, rx) = crossbeam_channel::bounded(1);
    saved.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    assert_eq!(&*saved.get_mapped_range(..).unwrap(), &[12; 4]);
    saved.unmap();
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn queued_submission_waiters_receive_the_first_device_failure() {
    let device = device(2 * 1024 * 1024);
    let queue = device.queue();
    let first = queue.reserve_submission().unwrap();
    let next = queue.reserve_submission().unwrap();
    let (tx, rx) = crossbeam_channel::bounded(1);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            tx.send(next.submit([]).unwrap_err()).unwrap();
        });
        assert!(rx.recv_timeout(Duration::from_millis(40)).is_err());
        device.runtime().fail(RenderFailure::new(
            RenderFailureKind::DeviceLost,
            "lost before stage submit",
        ));
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(2)).unwrap().kind(),
            RenderFailureKind::DeviceLost
        );
    });
    assert_eq!(
        first.submit([]).unwrap_err().kind(),
        RenderFailureKind::DeviceLost
    );
}
