use super::*;
use crate::gpu_runtime::{GpuBudgetConfig, GpuRuntime, OwnerKind, ResourceKind};
use std::time::Duration;

fn descriptors(staging: u64) -> Arc<Descriptors> {
    let backends = wgpu::Backends::VULKAN;
    let instance = crate::backend::create_wgpu_instance(backends, Default::default(), None);
    let (adapter, device, queue) =
        futures::executor::block_on(crate::backend::request_adapter_and_device(
            backends,
            &instance,
            None,
            wgpu::PowerPreference::HighPerformance,
        ))
        .unwrap();
    let mut budget = GpuBudgetConfig::for_adapter(adapter.get_info().device_type, false);
    budget.staging_limit = staging;
    let runtime = GpuRuntime::new(budget).unwrap();
    runtime.install_device_callbacks(&device);
    Arc::new(Descriptors::with_runtime(instance, adapter, device, queue, runtime).unwrap())
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn bitmap_uploads_exceeding_staging_budget_are_sliced_and_preserve_rows() {
    let descriptors = descriptors(4096);
    let mut context = WgpuContext3D::new(descriptors.clone(), Context3DProfile::Baseline).unwrap();
    // A readback-capable destination lets us verify this exact Stage3D command.
    let destination = Rc::new(TextureWrapper {
        texture: descriptors
            .device
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: 130,
                    height: 33,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                view_formats: &[],
                usage: TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
            })
            .unwrap(),
    });
    let source: Vec<u8> = (0..130 * 33 * 4).map(|i| (i % 251) as u8).collect();
    context
        .process_command(Context3DCommand::CopyBitmapToTexture {
            source: &source,
            source_width: 130,
            source_height: 33,
            dest: destination.clone(),
            layer: 0,
        })
        .unwrap();
    assert!(
        descriptors.runtime.stats().submitted > 0,
        "a mandatory upload larger than staging must submit completed slices"
    );
    let staging = descriptors.runtime.stats().kind_bytes[ResourceKind::StagingBuffer as usize];
    assert!(staging > 0 && staging <= 4096);
    context.buffer_staging_belt.finish();
    let mut encoder = std::mem::replace(
        &mut context.buffer_command_encoder,
        descriptors
            .device
            .create_command_encoder(&Default::default())
            .unwrap(),
    );
    let output = descriptors
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 768 * 33,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
        .unwrap();
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &destination.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(768),
                rows_per_image: None,
            },
        },
        Extent3d {
            width: 130,
            height: 33,
            depth_or_array_layers: 1,
        },
    );
    let submitted = descriptors
        .queue
        .submit([encoder.finish().unwrap()])
        .unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    output.map_async(wgpu::MapMode::Read, .., move |r| {
        tx.send(r).unwrap();
    });
    descriptors
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submitted),
            timeout: Some(Duration::from_secs(2)),
        })
        .unwrap();
    rx.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
    let bytes = output.get_mapped_range(..).unwrap();
    for row in 0..33 {
        assert_eq!(
            &bytes[row * 768..row * 768 + 520],
            &source[row * 520..(row + 1) * 520]
        );
    }
    drop(bytes);
    output.unmap();
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn surviving_stage3d_handles_are_not_misclassified_as_descriptor_resources() {
    let descriptors = descriptors(4096);
    let baseline: std::collections::HashSet<_> = descriptors
        .runtime
        .resources()
        .iter()
        .map(|r| r.id)
        .collect();
    let mut context = WgpuContext3D::new(descriptors.clone(), Context3DProfile::Baseline).unwrap();
    let texture = context
        .create_texture(8, 8, Context3DTextureFormat::Bgra, false, 0)
        .unwrap();
    drop(context);
    descriptors.runtime.purge_idle_and_cached_resources();
    let survivors: Vec<_> = descriptors
        .runtime
        .resources()
        .into_iter()
        .filter(|r| !baseline.contains(&r.id))
        .collect();
    assert!(!survivors.is_empty());
    assert!(
        survivors
            .iter()
            .all(|r| !matches!(r.owner.kind, OwnerKind::Renderer | OwnerKind::Host)),
        "live AS texture must remain attributable after its Context3D drops: {survivors:?}"
    );
    drop(texture);
    assert!(
        descriptors
            .runtime
            .resources()
            .iter()
            .all(|r| baseline.contains(&r.id))
    );
}
