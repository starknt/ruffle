use super::*;

// Keep a real transfer ahead of the uploads on the Vulkan queue. The zero-time
// fence check below proves the consumer is still busy at the measured peak.
fn slow_copy(device: &Device) -> (wgpu::CommandBuffer, [wgpu::Buffer; 2]) {
    let buffers = std::array::from_fn(|_| {
        device.raw().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16 << 20,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });
    let mut encoder = device.raw().create_command_encoder(&Default::default());
    for i in 0..512 {
        encoder.copy_buffer_to_buffer(
            &buffers[i % 2],
            0,
            &buffers[1 - i % 2],
            0,
            buffers[0].size(),
        );
    }
    (encoder.finish(), buffers)
}

#[test]
#[ignore = "requires Vulkan and wgpu/counters"]
fn slow_gpu_bounds_frame_and_queue_uploads() {
    const PAGE: u64 = 16384;
    const LIMIT: u64 = 4 * PAGE;
    const PIXELS: [u8; 65 * 4 * 32] = [11; 65 * 4 * 32];
    for submitters in [1, 2] {
        let device = device_with_staging_budget(1 << 20, LIMIT);
        let image = texture(&device, 65, 32, wgpu::TextureUsages::COPY_DST);
        let queue = device.queue();
        let layout = wgpu::TexelCopyBufferLayout {
            bytes_per_row: Some(260),
            ..Default::default()
        };
        let queue_write = || {
            queue
                .write_texture_observed(
                    image.as_image_copy(),
                    &PIXELS,
                    layout,
                    image.size(),
                    |payload, charge| {
                        assert_eq!((payload, charge), (8320, PAGE));
                    },
                )
                .unwrap()
        };
        let frame = || {
            let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
            let mut arena = StagingBelt::new(device.clone(), PAGE);
            arena
                .write_texture(
                    &mut encoder,
                    image.as_image_copy(),
                    &PIXELS,
                    false,
                    image.size(),
                )
                .unwrap();
            arena.finish();
            encoder.finish().unwrap()
        };
        let (slow, _buffers) = slow_copy(&device);
        queue.submit([]).unwrap();
        poll(&device);
        let baseline = hal_buffers(&device);
        let resident = device.runtime().stats().total_bytes;
        let slow = queue.raw().submit([slow]);
        for _ in 0..2 {
            let command = frame();
            queue_write();
            queue.submit([command]).unwrap();
        }
        let peak = staging(&device);
        let hal_peak = hal_buffers(&device) - baseline;
        assert_eq!(peak, LIMIT);
        assert!(
            hal_peak > 0 && hal_peak as u64 <= peak,
            "HAL buffers grew by {hal_peak}"
        );
        assert!(matches!(
            device.poll(wgpu::PollType::Wait {
                submission_index: Some(slow),
                timeout: Some(Duration::ZERO)
            }),
            Err(wgpu::PollError::Timeout)
        ));
        let start = std::sync::Barrier::new(submitters);
        std::thread::scope(|threads| {
            for i in 0..submitters {
                let start = &start;
                let frame = &frame;
                let queue_write = &queue_write;
                let queue = &queue;
                threads.spawn(move || {
                    start.wait();
                    if i == 0 {
                        queue_write();
                        assert!(queue.0.runtime().stats().completed >= 2);
                        queue.submit([]).unwrap();
                    } else {
                        let command = frame();
                        assert!(queue.0.runtime().stats().completed >= 2);
                        queue.submit([command]).unwrap();
                    }
                });
            }
        });
        poll(&device);
        device.runtime().purge_idle_and_cached_resources();
        let stats = device.runtime().stats();
        assert_eq!(stats.peak_bytes - resident, LIMIT);
        assert_eq!(stats.submitted, stats.completed);
        assert_eq!(staging(&device), 0);
        assert_eq!(hal_buffers(&device), baseline);
        eprintln!(
            "submitters={submitters} bytes staging={peak}/{LIMIT} HAL delta={hal_peak} drained staging=0 HAL delta=0 serials={}/{}",
            stats.completed, stats.submitted
        );
    }
}

#[test]
#[ignore = "requires Vulkan and wgpu/counters"]
fn oversized_padded_row_is_rejected_before_either_native_copy() {
    let device = device_with_staging_budget(16384, 300);
    let target = texture(&device, 65, 1, wgpu::TextureUsages::COPY_DST);
    let mut arena = StagingBelt::new(device.clone(), 256);
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    let baseline = hal_buffers(&device);
    // 260 useful bytes fit. The 512-byte padded row does not.
    let pixels = [0; 260];
    let error = arena
        .write_texture(
            &mut encoder,
            target.as_image_copy(),
            &pixels,
            false,
            target.size(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), RenderFailureKind::BudgetExceeded);
    let error = device
        .queue()
        .write_texture_observed(
            target.as_image_copy(),
            &pixels,
            Default::default(),
            target.size(),
            |_, _| panic!("unexpected native write"),
        )
        .unwrap_err();
    assert_eq!(error.kind(), RenderFailureKind::BudgetExceeded);
    assert_eq!(device.runtime().stats().submitted, 0);
    assert_eq!(device.queue().pending_upload_bytes(), 0);
    assert_eq!(staging(&device), 0);
    assert_eq!(hal_buffers(&device), baseline);
    device.runtime().check_health().unwrap();
}

fn texture(device: &Device, width: u32, height: u32, usage: wgpu::TextureUsages) -> Texture {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage,
            view_formats: &[],
        })
        .unwrap()
}

fn staging(device: &Device) -> u64 {
    device.runtime().stats().kind_bytes[ResourceKind::StagingBuffer as usize]
}

fn hal_buffers(device: &Device) -> isize {
    device.get_internal_counters().hal.buffer_memory.read()
}

fn poll(device: &Device) {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(3)),
        })
        .unwrap();
}

#[test]
#[ignore = "requires Vulkan and wgpu/counters"]
fn closed_stage_refunds_its_sealed_unsubmitted_page() {
    let device = device_with_staging_budget(16384, 768);
    let target = texture(&device, 1, 1, wgpu::TextureUsages::COPY_DST);
    let mut arena = StagingBelt::new(device.clone(), 256);
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    let baseline = hal_buffers(&device);
    arena
        .write_texture(
            &mut encoder,
            target.as_image_copy(),
            &[7; 4],
            false,
            target.size(),
        )
        .unwrap();
    arena.finish();
    let held = device.queue().reserve_submission().unwrap();
    let stage = encoder.into_stage_work().unwrap();
    device.runtime().purge_idle_and_cached_resources();
    assert_eq!(staging(&device), 256);
    let (close, closed) = crossbeam_channel::bounded(0);
    drop(close);
    assert_eq!(
        stage.submit_until_closed(&closed).unwrap_err().kind(),
        RenderFailureKind::Canceled
    );
    drop(held);
    device.runtime().purge_idle_and_cached_resources();
    poll(&device);
    assert_eq!(staging(&device), 0);
    assert_eq!(hal_buffers(&device), baseline);
    assert_eq!(device.runtime().stats().submitted, 0);
}

#[test]
#[ignore = "requires Vulkan and wgpu/counters"]
fn failed_native_write_drains_earlier_pending_uploads() {
    let device = device_with_staging_budget(16384, 4096);
    let valid = texture(&device, 64, 1, wgpu::TextureUsages::COPY_DST);
    let invalid = texture(&device, 64, 1, wgpu::TextureUsages::COPY_SRC);
    let queue = device.queue();
    queue.submit([]).unwrap();
    poll(&device);
    let baseline = hal_buffers(&device);
    let write = |target: &Texture| {
        queue.write_texture(
            target.as_image_copy(),
            &[5; 256],
            Default::default(),
            target.size(),
        )
    };
    write(&valid).unwrap();
    assert!(hal_buffers(&device) > baseline);
    let error = write(&invalid).unwrap_err();
    assert_eq!(error.kind(), RenderFailureKind::Validation);
    assert_eq!(device.runtime().failure(), Some(error));
    // The native progress worker stops on terminal failure.
    assert_eq!(staging(&device), 0);
    assert_eq!(queue.pending_upload_bytes(), 0);
    assert_eq!(hal_buffers(&device), baseline);
}

#[test]
#[ignore = "requires Vulkan and wgpu/counters"]
fn rejected_submission_can_drain_its_pending_native_writes() {
    let device = device_with_staging_budget(16384, 4096);
    let foreign = device_with_staging_budget(16384, 4096);
    let target = texture(&device, 64, 1, wgpu::TextureUsages::COPY_DST);
    let other = texture(&foreign, 64, 1, wgpu::TextureUsages::COPY_DST);
    let queue = device.queue();
    queue.submit([]).unwrap();
    poll(&device);
    let baseline = hal_buffers(&device);
    queue
        .write_texture(
            target.as_image_copy(),
            &[1; 256],
            Default::default(),
            target.size(),
        )
        .unwrap();
    assert!(hal_buffers(&device) > baseline);
    let error = queue
        .submit_external([], std::slice::from_ref(other.lease()))
        .unwrap_err();
    assert_eq!(error.kind(), RenderFailureKind::Validation);
    poll(&device);
    assert_eq!(hal_buffers(&device), baseline);
    assert_eq!(staging(&device), 0);
}

#[test]
#[ignore = "requires Vulkan"]
fn prefix_submit_leaves_another_submitters_mapped_view_writable() {
    let device = device_with_staging_budget(16384, 1536);
    let target = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
        .unwrap();
    let mut peer = StagingBelt::new(device.clone(), 256);
    let mut peer_encoder = device.create_command_encoder(&Default::default()).unwrap();
    let mut view = peer
        .write_buffer(
            &mut peer_encoder,
            &target,
            0,
            wgpu::BufferSize::new(16).unwrap(),
        )
        .unwrap();
    let image = texture(&device, 1, 4, wgpu::TextureUsages::COPY_DST);
    let mut arena = StagingBelt::new(device.clone(), 256);
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    arena
        .write_texture(
            &mut encoder,
            image.as_image_copy(),
            &[7; 16],
            false,
            image.size(),
        )
        .unwrap();
    assert!(device.runtime().stats().submitted > 0);
    view.copy_from_slice(&[3; 16]);
    drop(view);
    peer.finish();
    arena.finish();
    device
        .queue()
        .submit([encoder.finish().unwrap(), peer_encoder.finish().unwrap()])
        .unwrap();
    poll(&device);
    device.runtime().purge_idle_and_cached_resources();
    assert_eq!(staging(&device), 0);
}
