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

fn configure_back_buffer(context: &mut dyn Context3D, anti_alias: u32) {
    context
        .process_command(Context3DCommand::ConfigureBackBuffer {
            width: 2,
            height: 2,
            anti_alias,
            depth_and_stencil: true,
            wants_best_resolution: false,
            wants_best_resolution_on_browser_zoom: false,
        })
        .unwrap();
}

fn clear(context: &mut dyn Context3D, color: wgpu::Color, mask: u32) {
    context
        .process_command(Context3DCommand::Clear {
            red: color.r,
            green: color.g,
            blue: color.b,
            alpha: color.a,
            depth: 1.0,
            stencil: 0,
            mask,
        })
        .unwrap();
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn clear_without_draw_is_visible_in_the_presented_stage_packet() {
    use crate::{
        backend::WgpuRenderBackend, render_snapshot::bitmap_version, target::TextureTarget,
    };
    use ruffle_render::{
        backend::RenderBackend,
        commands::{CommandHandler, CommandList},
    };

    let descriptors = descriptors(4096);
    let target = TextureTarget::new(&descriptors.device, (2, 2)).unwrap();
    let mut renderer = WgpuRenderBackend::new(descriptors, target).unwrap();
    let mut context = renderer
        .create_context3d(Context3DProfile::Baseline)
        .unwrap();
    for anti_alias in [0, 4] {
        configure_back_buffer(&mut *context, anti_alias);
        let before = bitmap_version(&context.back_buffer_handle().unwrap()).unwrap();
        let prior_packet = renderer
            .capture_stage_packet(swf::Color::BLACK, CommandList::new(), vec![])
            .unwrap();
        clear(&mut *context, wgpu::Color::RED, COLOR_MASK);
        clear(&mut *context, wgpu::Color::BLUE, DEPTH_MASK | STENCIL_MASK);
        context.present();
        let front = context.bitmap_handle();
        let presented = bitmap_version(&front).unwrap();
        let mut stage = CommandList::new();
        stage.render_stage3d(front, Default::default());
        let packet = renderer
            .capture_stage_packet(swf::Color::BLACK, stage, vec![])
            .unwrap();
        assert!(packet.after_operation() >= prior_packet.after_operation() + 3);
        assert!(renderer.prepare_frame().unwrap());
        renderer.submit_stage_packet(packet).unwrap();
        assert_eq!(
            renderer.capture_frame().unwrap().unwrap().into_raw(),
            [255, 0, 0, 255].repeat(4)
        );
        assert_eq!(presented.key, before.key);
        assert_eq!(presented.version, before.version + 1);
    }
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn stage_render_scale_projects_the_presented_context3d_buffer_once() {
    use crate::{backend::WgpuRenderBackend, target::TextureTarget};
    use ruffle_render::{
        backend::RenderBackend,
        commands::{CommandHandler, CommandList},
        matrix::Matrix,
    };

    let descriptors = descriptors(4096);
    let target = TextureTarget::new(&descriptors.device, (8, 4)).unwrap();
    let mut renderer = WgpuRenderBackend::new(descriptors, target).unwrap();
    renderer.set_stage_render_scale_percent(50).unwrap();
    let mut context = renderer
        .create_context3d(Context3DProfile::Baseline)
        .unwrap();
    configure_back_buffer(&mut *context, 4);
    clear(&mut *context, wgpu::Color::RED, COLOR_MASK);
    context.present();
    let front = context.bitmap_handle();
    let texture = &crate::as_texture(&front, renderer.device())
        .unwrap()
        .texture;
    assert_eq!((texture.width(), texture.height()), (2, 2));
    let mut stage = CommandList::new();
    stage.render_stage3d(
        front,
        ruffle_render::transform::Transform {
            matrix: Matrix::translate(swf::Twips::from_pixels(4.), swf::Twips::from_pixels(2.)),
            ..Default::default()
        },
    );
    let packet = renderer
        .capture_stage_packet(swf::Color::BLACK, stage, vec![])
        .unwrap();
    assert!(renderer.prepare_frame().unwrap());
    renderer.submit_stage_packet(packet).unwrap();
    let image = renderer.capture_frame().unwrap().unwrap();
    assert_eq!(image.dimensions(), (4, 2));
    for (x, y, pixel) in image.enumerate_pixels() {
        assert_eq!(
            pixel.0,
            if (x, y) == (2, 1) {
                [255, 0, 0, 255]
            } else {
                [0, 0, 0, 255]
            }
        );
    }
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn offscreen_clear_survives_switching_back_to_the_back_buffer() {
    use crate::{backend::WgpuRenderBackend, target::TextureTarget};
    use ruffle_render::{backend::RenderBackend, bitmap::PixelRegion};
    use std::num::NonZeroU32;

    let descriptors = descriptors(4096);
    let mut context = WgpuContext3D::new(descriptors.clone(), Context3DProfile::Baseline).unwrap();
    let target = TextureTarget::new_without_readback(&descriptors.device, (2, 2)).unwrap();
    let mut renderer = WgpuRenderBackend::new(descriptors.clone(), target).unwrap();
    configure_back_buffer(&mut context, 0);
    let size = NonZeroU32::new(2).unwrap();
    // A readable target lets the test verify the actual offscreen pixels after the switch.
    let bitmap = renderer.create_empty_texture(size, size).unwrap();
    let texture = Rc::new(TextureWrapper {
        texture: crate::as_texture(&bitmap, &descriptors.device)
            .unwrap()
            .texture
            .clone(),
        _lifetime: context.operations.create_resource().unwrap(),
    });
    context
        .process_command(Context3DCommand::SetRenderToTexture {
            texture,
            enable_depth_and_stencil: false,
            anti_alias: 0,
            surface_selector: 0,
        })
        .unwrap();
    clear(&mut context, wgpu::Color::BLUE, COLOR_MASK);
    context
        .process_command(Context3DCommand::SetRenderToBackBuffer)
        .unwrap();
    clear(&mut context, wgpu::Color::RED, COLOR_MASK);
    context.present();
    renderer
        .read_bitmap_region(
            &bitmap,
            PixelRegion::for_pixel(0, 0),
            Box::new(|bytes, _| {
                assert_eq!(&bytes[..4], &[0, 0, 255, 255]);
            }),
        )
        .unwrap()
        .unwrap();
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn reconfigured_stage3d_buffer_keeps_its_identity_until_last_release() {
    use crate::render_snapshot::bitmap_version;
    let mut context = WgpuContext3D::new(descriptors(4096), Context3DProfile::Baseline).unwrap();
    for anti_alias in [0, 4] {
        configure_back_buffer(&mut context, anti_alias);
        let old = context.back_buffer_handle().unwrap();
        let previous = bitmap_version(&old).unwrap();
        configure_back_buffer(&mut context, anti_alias);
        let replacement = bitmap_version(&context.back_buffer_handle().unwrap()).unwrap();
        assert_ne!(replacement.key, previous.key);
        assert_eq!(bitmap_version(&old).unwrap(), previous);
        let configured = context.operations.applied();
        drop(old.clone());
        assert_eq!(context.operations.applied(), configured);
        drop(old);
        assert_eq!(context.operations.applied(), configured + 1);
    }
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn odd_index_upload_preserves_neighbors_and_snapshots_the_entire_input() {
    let descriptors = descriptors(4096);
    let mut context = WgpuContext3D::new(descriptors.clone(), Context3DProfile::Baseline).unwrap();
    let mut indices = IndexBufferWrapper {
        _lifetime: crate::render_snapshot::OperationSequence::default()
            .create_resource()
            .unwrap(),
        buffer: descriptors
            .device
            .create_buffer(&BufferDescriptor {
                label: None,
                size: 8,
                usage: BufferUsages::INDEX | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
            .unwrap(),
        data: vec![0; 8],
    };
    let mut bytes = [1, 2, 3, 4, 5, 6, 7, 8];
    context
        .process_command(Context3DCommand::UploadToIndexBuffer {
            buffer: &mut indices,
            start_offset: 0,
            data: &bytes,
        })
        .unwrap();
    bytes[..4].copy_from_slice(&[10, 11, 12, 13]);
    context
        .process_command(Context3DCommand::UploadToIndexBuffer {
            buffer: &mut indices,
            start_offset: 1,
            data: &bytes[..4],
        })
        .unwrap();
    bytes.fill(0);
    assert_eq!(
        read_buffer(&mut context, &indices.buffer),
        [1, 2, 10, 11, 12, 13, 7, 8]
    );
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn queued_stage3d_uploads_retain_bytes_and_execute_in_issue_order() {
    use super::uploads::{UploadLayout, UploadTarget};
    let descriptors = descriptors(4096);
    let mut context = WgpuContext3D::new(descriptors.clone(), Context3DProfile::Baseline).unwrap();
    let buffer = descriptors
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 8,
            usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
        .unwrap();
    let target = UploadTarget::Buffer(buffer.clone());
    let mut source = [1, 2, 3, 4, 5, 6, 7, 8];
    context
        .uploads
        .enqueue(target.clone(), UploadLayout::Buffer { offset: 0 }, &source)
        .unwrap();
    source[..4].copy_from_slice(&[9, 10, 11, 12]);
    context
        .uploads
        .enqueue(target, UploadLayout::Buffer { offset: 4 }, &source[..4])
        .unwrap();
    source.fill(0);
    context.flush_uploads().unwrap();
    assert_eq!(
        read_buffer(&mut context, &buffer),
        [1, 2, 3, 4, 9, 10, 11, 12]
    );
    assert_eq!(context.uploads.applied(), 2);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn saturated_upload_queue_leaves_sibling_uploads_and_pixel_reads_live() {
    use crate::{backend::WgpuRenderBackend, target::TextureTarget};
    use ruffle_render::{
        backend::RenderBackend,
        bitmap::{Bitmap, BitmapFormat, PixelRegion},
    };
    let descriptors = descriptors(4096);
    let context = || WgpuContext3D::new(descriptors.clone(), Context3DProfile::Baseline).unwrap();
    let mut stalled = context();
    let mut sibling = context();
    sibling.owned_uploads = true;
    let mut renderer = WgpuRenderBackend::with_options(
        descriptors.clone(),
        TextureTarget::new(&descriptors.device, (2, 1)).unwrap(),
        false,
    )
    .unwrap();
    let bitmap = renderer
        .register_bitmap(Bitmap::new(
            2,
            1,
            BitmapFormat::Rgba,
            vec![2, 4, 6, 255, 23, 67, 211, 255],
        ))
        .unwrap();
    let buffer = descriptors
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: 4096,
            usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
        .unwrap();
    let target = UploadTarget::Buffer(buffer.clone());
    let layout = UploadLayout::Buffer { offset: 0 };
    stalled
        .uploads
        .enqueue(target.clone(), layout, &[0; 4096])
        .unwrap();
    let full = sibling
        .uploads
        .enqueue(target.clone(), layout, &[0; 4])
        .unwrap_err();
    assert_eq!(
        full.kind(),
        crate::gpu_runtime::RenderFailureKind::BudgetExceeded
    );

    let mut indices = IndexBufferWrapper {
        _lifetime: sibling.operations.create_resource().unwrap(),
        buffer: buffer.clone(),
        data: vec![0; 4096],
    };
    sibling
        .process_command(Context3DCommand::UploadToIndexBuffer {
            buffer: &mut indices,
            start_offset: 0,
            data: &[11, 22, 33, 44],
        })
        .unwrap();
    assert_eq!(&read_buffer(&mut sibling, &buffer)[..4], &[11, 22, 33, 44]);
    let mut observed = Vec::new();
    renderer
        .read_bitmap_region(
            &bitmap,
            PixelRegion::for_pixel(1, 0),
            Box::new(|pixels, _| observed.extend_from_slice(&pixels[..4])),
        )
        .unwrap()
        .unwrap();
    assert_eq!(observed, [23, 67, 211, 255]);
    drop(stalled);
    sibling.uploads.enqueue(target, layout, &[0; 4]).unwrap();
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn large_stage3d_vertex_upload_preserves_every_byte_across_staging_pages() {
    let descriptors = descriptors(4096);
    let mut context = WgpuContext3D::new(descriptors.clone(), Context3DProfile::Baseline).unwrap();
    context.owned_uploads = true;
    let source: Vec<u8> = (0..12292).map(|i| (i % 251) as u8).collect();
    let buffer = descriptors
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: source.len() as u64,
            usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::VERTEX,
            mapped_at_creation: false,
        })
        .unwrap();
    let vertex = Rc::new(VertexBufferWrapper {
        _lifetime: context.operations.create_resource().unwrap(),
        buffer: buffer.clone(),
        data_32_per_vertex: 1,
    });
    context
        .process_command(Context3DCommand::UploadToVertexBuffer {
            buffer: vertex,
            start_vertex: 0,
            data32_per_vertex: 1,
            data: &source,
        })
        .unwrap();
    assert_eq!(read_buffer(&mut context, &buffer), source);
}

fn read_buffer(context: &mut WgpuContext3D, source: &crate::gpu_resources::Buffer) -> Vec<u8> {
    let descriptors = context.descriptors.clone();
    let output = descriptors
        .device
        .create_buffer(&BufferDescriptor {
            label: None,
            size: source.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
        .unwrap();
    context.buffer_staging_belt.finish();
    let mut encoder = std::mem::replace(
        &mut context.buffer_command_encoder,
        descriptors
            .device
            .create_command_encoder(&Default::default())
            .unwrap(),
    );
    encoder.copy_buffer_to_buffer(source, 0, &output, 0, source.size());
    let index = descriptors
        .queue
        .submit([encoder.finish().unwrap()])
        .unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    output.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap()
    });
    descriptors
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::from_secs(3)),
        })
        .unwrap();
    rx.recv_timeout(Duration::from_secs(3)).unwrap().unwrap();
    let bytes = output.get_mapped_range(..).unwrap().to_vec();
    output.unmap();
    bytes
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn bitmap_uploads_exceeding_staging_budget_are_sliced_and_preserve_rows() {
    let descriptors = descriptors(4096);
    let mut context = WgpuContext3D::new(descriptors.clone(), Context3DProfile::Baseline).unwrap();
    // A readback-capable destination lets us verify this exact Stage3D command.
    let destination = Rc::new(TextureWrapper {
        _lifetime: context.operations.create_resource().unwrap(),
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
    let bgra: Vec<u8> = source
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&[r, g, b, a]| [b, g, r, a])
        .collect();
    context
        .process_command(Context3DCommand::CopyBitmapToTexture {
            source: &bgra,
            source_is_bgra: true,
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
