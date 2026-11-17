//! Real-device regressions; enable wgpu/counters when running these.
use crate::backend::{WgpuRenderBackend, request_adapter_and_device};
use crate::descriptors::Descriptors;
use crate::target::RenderTarget;
use crate::target::TextureTarget;
use ruffle_render::backend::RenderBackend;
use ruffle_render::commands::{CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::matrix::Matrix;
use ruffle_render::quality::StageQuality;
use std::sync::Arc;
use swf::{BlendMode, Color, Twips};

#[test]
#[ignore = "requires a Vulkan adapter"]
fn measurement_target_never_allocates_a_readback_buffer_even_after_resize() {
    let descriptors = gpu();
    let mut target = TextureTarget::new_without_readback(&descriptors.device, (32, 32)).unwrap();
    assert!(target.buffer.is_none());
    target
        .resize(&descriptors.device, 64, 48)
        .expect("GPU operation failed");
    assert!(target.buffer.is_none());
    assert_eq!((target.width(), target.height()), (64, 48));
    let mut renderer = WgpuRenderBackend::with_options(descriptors, target, false).unwrap();
    renderer
        .submit_frame(Color::RED, CommandList::new(), vec![])
        .expect("GPU operation failed");
    assert!(renderer.capture_frame().unwrap().is_none());
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn measurement_counts_encoded_draws_and_passes() {
    let descriptors = gpu();
    let target = TextureTarget::new_without_readback(&descriptors.device, (32, 32)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors, target, false).unwrap();
    let metrics = ruffle_render::profile::RenderMetrics::default();
    let _scope = ruffle_render::profile::enter(Some(&metrics));
    let mut commands = CommandList::new();
    rect(&mut commands, 0.0, 0.0, 10.0, 10.0, Color::RED);
    renderer
        .submit_frame(Color::BLACK, commands, vec![])
        .expect("GPU operation failed");
    let work = metrics.snapshot().work;
    // One direct shape pass, with no surface-copy or capture pass.
    assert_eq!(work.draw_calls, 1);
    assert_eq!(work.render_passes, 1);
    assert_eq!(work.copy_bytes, 0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn measurement_counts_mask_passes_not_mask_inputs() {
    let descriptors = gpu();
    let target = TextureTarget::new_without_readback(&descriptors.device, (32, 32)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors, target, false).unwrap();
    let metrics = ruffle_render::profile::RenderMetrics::default();
    let _scope = ruffle_render::profile::enter(Some(&metrics));
    let mut source = CommandList::new();
    rect(&mut source, 2.0, 2.0, 12.0, 12.0, Color::RED);
    let mut mask = CommandList::new();
    rect(&mut mask, 2.0, 2.0, 12.0, 12.0, Color::WHITE);
    let mut commands = CommandList::new();
    commands.render_alpha_mask(source, mask);
    renderer
        .submit_frame(Color::BLACK, commands, vec![])
        .expect("GPU operation failed");
    assert_eq!(metrics.snapshot().work.mask_passes, 1);
    assert_eq!(metrics.snapshot().work.offscreen_targets, 2);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn bounded_compositing_matches_full_target_pixels() {
    let descriptors = gpu();
    for quality in [StageQuality::Low, StageQuality::High] {
        for alpha_mask in [false, true] {
            let mut results = Vec::new();
            for full in [true, false] {
                let target = TextureTarget::new(&descriptors.device, (128, 128)).unwrap();
                let mut renderer =
                    WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
                renderer.set_quality(quality).expect("GPU operation failed");
                let metrics = ruffle_render::profile::RenderMetrics::default();
                let _scope = ruffle_render::profile::enter(Some(&metrics));
                let mut foreground = CommandList::new();
                if full {
                    rect(&mut foreground, 0.0, 0.0, 128.0, 128.0, Color::TRANSPARENT);
                }
                rect(
                    &mut foreground,
                    37.25,
                    19.5,
                    13.5,
                    21.25,
                    Color::from_rgb(0x63a1ce, 193),
                );
                let mut commands = CommandList::new();
                if alpha_mask {
                    let mut mask = CommandList::new();
                    rect(
                        &mut mask,
                        32.0,
                        22.25,
                        16.0,
                        28.0,
                        Color::from_rgb(0xffffff, 127),
                    );
                    commands.render_alpha_mask(foreground, mask);
                } else {
                    // Nested origins must compose correctly as well.
                    let mut inner = CommandList::new();
                    inner.blend(foreground, RenderBlendMode::Builtin(BlendMode::Multiply));
                    commands.blend(inner, RenderBlendMode::Builtin(BlendMode::Layer));
                }
                renderer
                    .submit_frame(Color::from_rgb(0x96b052, 255), commands, vec![])
                    .expect("GPU operation failed");
                let pixels = renderer
                    .capture_frame()
                    .expect("GPU operation failed")
                    .unwrap()
                    .into_raw();
                results.push((pixels, metrics.snapshot().work));
            }
            assert_eq!(
                results[0].0, results[1].0,
                "quality {quality:?}, mask {alpha_mask}"
            );
            assert!(results[1].1.offscreen_pixels * 4 < results[0].1.offscreen_pixels);
            if !alpha_mask {
                // Capture copies the whole output once in both runs.
                let capture = 128 * 128 * 4;
                assert!(
                    (results[1].1.copy_bytes - capture) * 4 < results[0].1.copy_bytes - capture
                );
            }
        }
    }
}

fn gpu() -> Arc<Descriptors> {
    gpu_with_watermark(None)
}

fn gpu_with_watermark(watermark: Option<u64>) -> Arc<Descriptors> {
    let mut options = wgpu::InstanceDescriptor::new_without_display_handle();
    options.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(options);
    let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
        wgpu::Backends::VULKAN,
        &instance,
        None,
        wgpu::PowerPreference::HighPerformance,
    ))
    .expect("explicit GPU regressions require a Vulkan adapter");
    eprintln!("native GPU regression adapter: {:?}", adapter.get_info());
    let mut config =
        crate::gpu_runtime::GpuBudgetConfig::for_adapter(adapter.get_info().device_type, false);
    if let Some(watermark) = watermark {
        config.transient_watermark = watermark;
    }
    let runtime = crate::gpu_runtime::GpuRuntime::new(config).unwrap();
    runtime.install_device_callbacks(&device);
    Arc::new(
        Descriptors::with_runtime(instance, adapter, device, queue, runtime)
            .expect("GPU operation failed"),
    )
}

fn rect(list: &mut CommandList, x: f32, y: f32, w: f32, h: f32, color: Color) {
    list.draw_rect(
        color,
        Matrix {
            a: w,
            b: 0.0,
            c: 0.0,
            d: h,
            tx: Twips::from_pixels(x.into()),
            ty: Twips::from_pixels(y.into()),
        },
    );
}

fn scene(frame: usize, masked: bool, parent_alpha: bool) -> CommandList {
    let mut list = CommandList::new();
    if masked {
        list.push_mask();
        rect(&mut list, 8.0, 8.0, 96.0, 96.0, Color::WHITE);
        list.activate_mask();
    }
    for i in 0..96 {
        if masked && i == 32 {
            list.push_mask();
            rect(&mut list, 24.0, 24.0, 64.0, 64.0, Color::WHITE);
            list.activate_mask();
        } else if masked && i == 64 {
            list.deactivate_mask();
            rect(&mut list, 24.0, 24.0, 64.0, 64.0, Color::WHITE);
            list.pop_mask();
        }
        let mut child = CommandList::new();
        let color = Color::from_rgb(
            if (i + frame) % 2 == 0 {
                0x248fbd
            } else {
                0xbd4024
            },
            200,
        );
        rect(
            &mut child,
            4.0 + (i % 8) as f32 * 10.0,
            4.0 + (i / 8) as f32 * 7.0,
            24.0,
            24.0,
            color,
        );
        if parent_alpha && i % 7 == 0 {
            let mut alpha = CommandList::new();
            rect(
                &mut alpha,
                12.0,
                12.0,
                64.0,
                64.0,
                Color::from_rgb(0xffffff, 128),
            );
            child.blend(
                alpha,
                RenderBlendMode::Builtin(if i % 2 == 0 {
                    BlendMode::Alpha
                } else {
                    BlendMode::Erase
                }),
            );
        }
        if i % 5 == 0 {
            let mut mask = CommandList::new();
            rect(
                &mut mask,
                4.0,
                4.0,
                100.0,
                100.0,
                Color::from_rgb(0xffffff, 128),
            );
            list.render_alpha_mask(child, mask);
        } else {
            let mode = [
                BlendMode::Layer,
                BlendMode::Screen,
                BlendMode::Multiply,
                BlendMode::Subtract,
            ][i % 4];
            list.blend(child, RenderBlendMode::Builtin(mode));
        }
    }
    if masked {
        list.deactivate_mask();
        rect(&mut list, 8.0, 8.0, 96.0, 96.0, Color::WHITE);
        list.pop_mask();
    }
    // Known pixels independent of the implementation being compared.
    rect(&mut list, 116.0, 116.0, 8.0, 8.0, Color::RED);
    list
}

fn render(
    descriptors: &Arc<Descriptors>,
    masked: bool,
    parent_alpha: bool,
    quality: StageQuality,
) -> (Vec<Vec<u8>>, isize) {
    let before = descriptors
        .device
        .get_internal_counters()
        .hal
        .texture_memory
        .read();
    let target = TextureTarget::new(&descriptors.device, (128, 128)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
    renderer.set_quality(quality).expect("GPU operation failed");
    let mut frames = Vec::new();
    let mut live = 0;
    for frame in 0..3 {
        renderer
            .submit_frame(Color::WHITE, scene(frame, masked, parent_alpha), vec![])
            .expect("GPU operation failed");
        let image = renderer
            .capture_frame()
            .expect("GPU operation failed")
            .expect("rendered frame must be readable");
        assert_eq!(image.get_pixel(0, 0).0, [255, 255, 255, 255]);
        assert_eq!(image.get_pixel(120, 120).0, [255, 0, 0, 255]);
        frames.push(image.into_raw());
        live = live.max(
            descriptors
                .device
                .get_internal_counters()
                .hal
                .texture_memory
                .read()
                - before,
        );
    }
    assert!(
        live > 0,
        "missing live texture bytes (run with wgpu/counters): before={before}, counters={:?}",
        descriptors.device.get_internal_counters()
    );
    drop(renderer);
    descriptors
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    assert_eq!(
        descriptors
            .device
            .get_internal_counters()
            .hal
            .texture_memory
            .read(),
        before,
        "renderer teardown leaked texture allocations"
    );
    (frames, live)
}

#[test]
#[ignore = "requires a real Vulkan adapter"]
fn blend_batches_are_enabled_by_default() {
    let descriptors = gpu();
    let target = TextureTarget::new(&descriptors.device, (128, 128)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
    let metrics = ruffle_render::profile::RenderMetrics::default();
    let _scope = ruffle_render::profile::enter(Some(&metrics));
    renderer
        .submit_frame(Color::WHITE, scene(0, true, false), vec![])
        .expect("GPU operation failed");
    let expected = renderer
        .capture_frame()
        .expect("GPU operation failed")
        .expect("default renderer capture");
    assert!(
        metrics.snapshot().blend_batch_splits > 0,
        "bounded batching must be enabled by the renderer default"
    );
    renderer
        .set_viewport_dimensions(renderer.viewport_dimensions())
        .unwrap();
    let before = metrics.snapshot().blend_batch_splits;
    renderer
        .submit_frame(Color::WHITE, scene(0, true, false), vec![])
        .expect("GPU operation failed");
    let actual = renderer
        .capture_frame()
        .expect("GPU operation failed")
        .expect("configured renderer capture");
    assert_eq!(actual, expected, "batch policy changed pixels");
    assert!(metrics.snapshot().blend_batch_splits > before);
}

#[test]
#[ignore = "requires a real Vulkan adapter and wgpu/counters"]
fn blend_batches_reuse_textures_and_preserve_masked_pixels() {
    let descriptors = gpu();
    for quality in [StageQuality::Low, StageQuality::High] {
        for masked in [false, true] {
            let (frames, live) = render(&descriptors, masked, false, quality);
            eprintln!("blendTemporaryBytes quality={quality:?} masked={masked} live={live}");
            assert!(!frames.is_empty());
        }
    }
}

#[test]
#[ignore = "requires a real Vulkan adapter and wgpu/counters"]
fn ancestor_alpha_and_erase_preserve_original_layer_sampling() {
    let descriptors = gpu();
    let (frames, _) = render(&descriptors, true, true, StageQuality::High);
    assert!(!frames.is_empty());
}

#[test]
#[ignore = "optional Vulkan benchmark; emits measurements without a timing threshold"]
fn benchmark_blend_batches_encode_and_submit() {
    let descriptors = gpu();
    let target = TextureTarget::new(&descriptors.device, (128, 128)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
    renderer
        .set_quality(StageQuality::High)
        .expect("GPU operation failed");
    let mut timings = Vec::new();
    for frame in 0..34 {
        let commands = scene(frame, true, false);
        let start = std::time::Instant::now();
        renderer
            .submit_frame(Color::WHITE, commands, vec![])
            .expect("GPU operation failed");
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        renderer
            .capture_frame()
            .expect("GPU operation failed")
            .expect("benchmark capture failed");
        if frame >= 3 {
            timings.push(elapsed);
        }
    }
    timings.sort_by(f64::total_cmp);
    println!(
        "blendBatchBenchmark optimized=true samples={} encodeSubmitWallP50Ms={:.3} encodeSubmitWallP95Ms={:.3}",
        timings.len(),
        timings[15],
        timings[29]
    );
    drop(renderer);
    descriptors
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

#[test]
#[ignore = "requires a real Vulkan adapter"]
fn immutable_bitmap_eviction_preserves_recorded_pixels_and_releases_gpu_memory() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat, PixelRegion, PixelSnapping};
    use ruffle_render::transform::Transform;

    let descriptors = gpu();
    let memory = || {
        descriptors
            .device
            .get_internal_counters()
            .hal
            .texture_memory
            .read()
    };
    let target = TextureTarget::new(&descriptors.device, (128, 128)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
    renderer
        .set_quality(StageQuality::Low)
        .expect("GPU operation failed");
    renderer
        .submit_frame(Color::WHITE, CommandList::new(), vec![])
        .expect("GPU operation failed");
    renderer
        .capture_frame()
        .expect("GPU operation failed")
        .unwrap();
    let baseline = memory();
    assert!(baseline > 0, "run with wgpu/counters");
    // Retain exactly one upload. The first draw must survive eviction before
    // submit_frame, and subsequent reuploads must preserve independent pixels.
    renderer.set_immutable_bitmap_cache_limit(Some(64 * 64 * 4));
    for _ in 0..3 {
        let mut commands = CommandList::new();
        for (x, rgba) in [(0.0, [255, 0, 0, 255]), (64.0, [0, 0, 255, 255])] {
            let handle = renderer
                .register_bitmap(Bitmap::new(
                    64,
                    64,
                    BitmapFormat::Rgba,
                    rgba.repeat(64 * 64),
                ))
                .unwrap();
            assert!(renderer.cache_immutable_bitmap(&handle));
            commands.render_bitmap(
                handle,
                Transform {
                    matrix: Matrix::translate(Twips::from_pixels(x), Twips::ZERO),
                    ..Default::default()
                },
                false,
                PixelSnapping::Never,
                PixelRegion::for_region(0, 0, 64, 64),
            );
        }
        renderer
            .submit_frame(Color::WHITE, commands, vec![])
            .expect("GPU operation failed");
        let actual = renderer
            .capture_frame()
            .expect("GPU operation failed")
            .unwrap();
        assert_eq!(actual.get_pixel(32, 32).0, [255, 0, 0, 255]);
        assert_eq!(actual.get_pixel(96, 32).0, [0, 0, 255, 255]);
        assert_eq!(actual.get_pixel(32, 96).0, [255, 255, 255, 255]);
    }
    // Drain only in this explicit test, never in the eviction implementation.
    renderer
        .submit_frame(Color::WHITE, CommandList::new(), vec![])
        .expect("GPU operation failed");
    renderer
        .capture_frame()
        .expect("GPU operation failed")
        .unwrap();
    descriptors
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let retained = memory();
    renderer.set_immutable_bitmap_cache_limit(None);
    descriptors
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    assert_eq!(
        retained - memory(),
        64 * 64 * 4,
        "only one cached upload should remain"
    );
    assert_eq!(memory(), baseline, "evicted native GPU textures leaked");
}

#[test]
#[ignore = "requires a real Vulkan adapter"]
fn immutable_bitmap_cache_is_enabled_by_default() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat};
    let descriptors = gpu();
    let target = TextureTarget::new(&descriptors.device, (4, 4)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
    let handle = renderer
        .register_bitmap(Bitmap::new(1, 1, BitmapFormat::Rgba, vec![255; 4]))
        .unwrap();
    assert!(renderer.cache_immutable_bitmap(&handle));
    renderer.set_immutable_bitmap_cache_limit(Some(4));
    assert!(renderer.cache_immutable_bitmap(&handle));
    renderer.set_immutable_bitmap_cache_limit(None);
    assert!(!renderer.cache_immutable_bitmap(&handle));
}

#[test]
#[ignore = "requires a real Vulkan adapter"]
fn bitmap_gpu_copy_preserves_the_requested_region() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat, PixelRegion};

    let descriptors = gpu();
    let target = TextureTarget::new(&descriptors.device, (1, 1)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors, target, false).unwrap();

    let source_pixels = vec![
        0x10, 0x20, 0x30, 0x40, // source (0, 0)
        0x50, 0x60, 0x70, 0x80, // source (1, 0)
        0x90, 0xA0, 0xB0, 0xC0, // source (0, 1)
        0xD0, 0xE0, 0xF0, 0xFF, // source (1, 1)
    ];
    let source = renderer
        .register_bitmap(Bitmap::new(2, 2, BitmapFormat::Rgba, source_pixels.clone()))
        .unwrap();
    let destination = renderer
        .register_bitmap(Bitmap::new(4, 4, BitmapFormat::Rgba, vec![0; 4 * 4 * 4]))
        .unwrap();

    let sync = renderer
        .copy_bitmap_region(
            source,
            PixelRegion::for_region(0, 0, 2, 2),
            destination,
            PixelRegion::for_region(1, 1, 2, 2),
        )
        .unwrap()
        .expect("the wgpu backend should provide a GPU copy sync handle");

    let mut copied = Vec::new();
    renderer
        .resolve_sync_handle(
            sync,
            Box::new(|bytes, row_pitch| {
                for row in 0..2 {
                    let start = row as usize * row_pitch as usize;
                    copied.extend_from_slice(&bytes[start..start + 2 * 4]);
                }
            }),
        )
        .unwrap();

    assert_eq!(copied, source_pixels);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn destroyed_readback_buffer_returns_error_without_publishing_pixels()
-> Result<(), crate::gpu_runtime::RenderFailure> {
    let descriptors = gpu();
    let dimensions = crate::utils::BufferDimensions::new(1, 1, wgpu::TextureFormat::Rgba8Unorm);
    let buffer = descriptors.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("injected map failure"),
        size: 256,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })?;
    buffer.destroy();
    let published = std::cell::Cell::new(false);
    let result =
        crate::utils::capture_image(&descriptors.device, &buffer, &dimensions, None, |_, _| {
            published.set(true)
        });
    assert!(result.is_err());
    assert!(!published.get());

    Ok(())
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn cheap_operations_do_not_force_a_draw_count_submission() {
    let descriptors = gpu();
    let budget = crate::idle_pool::Budget::with_runtime(&descriptors.runtime);
    let mut frame = crate::backend::ActiveFrame::new(&descriptors, budget).unwrap();
    let buffer = descriptors
        .device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
        .unwrap();
    for _ in 0..120 {
        frame.encoder().clear_buffer(&buffer, 0, None);
        frame.maybe_flush(&descriptors).unwrap();
    }
    assert_eq!(descriptors.runtime.stats().submitted, 0);
    frame.submit_direct(&descriptors).unwrap();
    assert_eq!(descriptors.runtime.stats().submitted, 1);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn transient_bytes_trigger_submission_at_an_operation_boundary() {
    let descriptors = gpu_with_watermark(Some(64));
    let budget = crate::idle_pool::Budget::with_runtime(&descriptors.runtime);
    let mut frame = crate::backend::ActiveFrame::new(&descriptors, budget).unwrap();
    let texture = descriptors
        .device
        .create_texture_with_kind(
            &wgpu::TextureDescriptor {
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
            },
            crate::gpu_runtime::ResourceKind::OffscreenTarget,
        )
        .unwrap();
    let view = texture.create_view(&Default::default()).unwrap();
    {
        let _pass = frame
            .encoder()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
    }
    frame.maybe_flush(&descriptors).unwrap();
    assert_eq!(descriptors.runtime.stats().submitted, 1);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn bitmap_registration_is_lazy_and_first_visible_use_preserves_pixels() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat, PixelRegion, PixelSnapping};
    use ruffle_render::transform::Transform;
    let descriptors = gpu();
    let target = TextureTarget::new(&descriptors.device, (8, 8)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
    let before = descriptors.runtime.stats().total_bytes;
    let handle = renderer
        .register_bitmap(Bitmap::new(
            8,
            8,
            BitmapFormat::Rgba,
            [255, 0, 0, 255].repeat(64),
        ))
        .unwrap();
    assert_eq!(
        descriptors.runtime.stats().total_bytes,
        before,
        "invisible registration cannot allocate or upload on the GPU"
    );
    let mut commands = CommandList::new();
    commands.render_bitmap(
        handle,
        Transform::default(),
        false,
        PixelSnapping::Never,
        PixelRegion::for_whole_size(8, 8),
    );
    renderer
        .submit_frame(Color::BLACK, commands, vec![])
        .unwrap();
    let image = renderer.capture_frame().unwrap().unwrap();
    assert!(image.pixels().all(|p| p.0 == [255, 0, 0, 255]));
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn deferred_partial_uploads_preserve_unwritten_and_gpu_written_pixels() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat, PixelRegion, PixelSnapping};
    use ruffle_render::transform::Transform;
    let descriptors = gpu();
    let target = TextureTarget::new(&descriptors.device, (4, 4)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors, target, false).unwrap();
    let pixels = |color: [u8; 4]| Bitmap::new(4, 4, BitmapFormat::Rgba, color.repeat(16));
    let handle = renderer.register_bitmap(pixels([255, 0, 0, 255])).unwrap();
    renderer
        .update_texture(
            &handle,
            pixels([0, 0, 255, 255]),
            PixelRegion::for_pixel(1, 1),
        )
        .unwrap();
    let draw = |renderer: &mut WgpuRenderBackend<TextureTarget>| {
        let mut commands = CommandList::new();
        commands.render_bitmap(
            handle.clone(),
            Transform::default(),
            false,
            PixelSnapping::Never,
            PixelRegion::for_whole_size(4, 4),
        );
        renderer
            .submit_frame(Color::BLACK, commands, vec![])
            .unwrap();
        renderer.capture_frame().unwrap().unwrap()
    };
    let image = draw(&mut renderer);
    assert_eq!(image.get_pixel(1, 1).0, [0, 0, 255, 255]);
    assert_eq!(image.get_pixel(2, 2).0, [255, 0, 0, 255]);
    let green = renderer.register_bitmap(pixels([0, 255, 0, 255])).unwrap();
    renderer
        .copy_bitmap_region(
            green,
            PixelRegion::for_pixel(0, 0),
            handle.clone(),
            PixelRegion::for_pixel(2, 2),
        )
        .unwrap()
        .unwrap();
    renderer
        .update_texture(
            &handle,
            pixels([255, 255, 255, 255]),
            PixelRegion::for_pixel(0, 0),
        )
        .unwrap();
    renderer
        .update_texture(
            &handle,
            pixels([0, 0, 0, 255]),
            PixelRegion::for_pixel(3, 3),
        )
        .unwrap();
    let image = draw(&mut renderer);
    assert_eq!(image.get_pixel(0, 0).0, [255, 255, 255, 255]);
    assert_eq!(image.get_pixel(3, 3).0, [0, 0, 0, 255]);
    assert_eq!(image.get_pixel(2, 2).0, [0, 255, 0, 255]);
    assert_eq!(image.get_pixel(1, 1).0, [0, 0, 255, 255]);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn shape_registration_owns_styles_and_allocates_only_when_visible() {
    use ruffle_render::{
        bitmap::{BitmapHandle, BitmapSize, BitmapSource},
        shape_utils::{DistilledShape, DrawCommand, DrawPath, FillRule},
        transform::Transform,
    };
    struct NoBitmaps;
    impl BitmapSource for NoBitmaps {
        fn bitmap_size(&self, _: u16) -> Option<BitmapSize> {
            None
        }
        fn bitmap_handle(&self, _: u16, _: &mut dyn RenderBackend) -> Option<BitmapHandle> {
            None
        }
    }
    let descriptors = gpu();
    let target = TextureTarget::new(&descriptors.device, (8, 8)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
    let before = descriptors.runtime.stats().total_bytes;
    let handle = {
        let style = swf::FillStyle::Color(Color::RED);
        let point = |x, y| swf::Point::new(Twips::from_pixels(x), Twips::from_pixels(y));
        renderer
            .register_shape(
                DistilledShape {
                    id: 7,
                    shape_bounds: Default::default(),
                    edge_bounds: Default::default(),
                    paths: vec![DrawPath::Fill {
                        style: &style,
                        winding_rule: FillRule::EvenOdd,
                        commands: vec![
                            DrawCommand::MoveTo(point(0.0, 0.0)),
                            DrawCommand::LineTo(point(8.0, 0.0)),
                            DrawCommand::LineTo(point(8.0, 8.0)),
                            DrawCommand::LineTo(point(0.0, 8.0)),
                            DrawCommand::LineTo(point(0.0, 0.0)),
                        ],
                    }],
                },
                &NoBitmaps,
            )
            .unwrap()
    };
    assert_eq!(descriptors.runtime.stats().total_bytes, before);
    let mut commands = CommandList::new();
    commands.render_shape(handle, Transform::default());
    renderer
        .submit_frame(Color::BLACK, commands, vec![])
        .unwrap();
    let image = renderer.capture_frame().unwrap().unwrap();
    assert!(image.pixels().all(|p| p.0 == [255, 0, 0, 255]));
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn numeric_snapshot_freezes_pixels_and_rejects_a_changed_version_before_allocation() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat, PixelRegion, PixelSnapping};
    use ruffle_render::transform::Transform;
    fn send_sync<T: Send + Sync>() {}
    send_sync::<crate::render_snapshot::RenderSnapshot>();
    let descriptors = gpu();
    let target = TextureTarget::new(&descriptors.device, (4, 4)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
    let handle = renderer
        .register_bitmap(Bitmap::new(
            4,
            4,
            BitmapFormat::Rgba,
            [255, 0, 0, 255].repeat(16),
        ))
        .unwrap();
    let before = descriptors.runtime.stats().total_bytes;
    let mut commands = CommandList::new();
    commands.render_bitmap(
        handle.clone(),
        Transform::default(),
        false,
        PixelSnapping::Never,
        PixelRegion::for_whole_size(4, 4),
    );
    let frozen = crate::render_snapshot::PreparedSnapshot::capture(&commands).unwrap();
    renderer
        .update_texture(
            &handle,
            Bitmap::new(4, 4, BitmapFormat::Rgba, [0, 0, 255, 255].repeat(16)),
            PixelRegion::for_whole_size(4, 4),
        )
        .unwrap();
    let crate::render_snapshot::ResourceRecipe::Bitmap { pixels, .. } = &frozen.snapshot.recipes[0]
    else {
        panic!("expected owned pixel recipe");
    };
    assert_eq!(&pixels.data()[..4], &[255, 0, 0, 255]);
    assert!(frozen.resolve(&mut renderer).is_err());
    assert_eq!(descriptors.runtime.stats().total_bytes, before);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn numeric_snapshot_rejects_bitmap_changed_by_gpu_copy() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat, PixelRegion, PixelSnapping};
    use ruffle_render::transform::Transform;
    let descriptors = gpu();
    let target = TextureTarget::new(&descriptors.device, (4, 4)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors, target, false).unwrap();
    let red = renderer
        .register_bitmap(Bitmap::new(
            4,
            4,
            BitmapFormat::Rgba,
            [255, 0, 0, 255].repeat(16),
        ))
        .unwrap();
    let blue = renderer
        .register_bitmap(Bitmap::new(
            4,
            4,
            BitmapFormat::Rgba,
            [0, 0, 255, 255].repeat(16),
        ))
        .unwrap();
    let mut commands = CommandList::new();
    commands.render_bitmap(
        red.clone(),
        Transform::default(),
        false,
        PixelSnapping::Never,
        PixelRegion::for_whole_size(4, 4),
    );
    let frozen = crate::render_snapshot::PreparedSnapshot::capture(&commands).unwrap();
    renderer
        .copy_bitmap_region(
            blue,
            PixelRegion::for_pixel(0, 0),
            red,
            PixelRegion::for_pixel(0, 0),
        )
        .unwrap()
        .unwrap();
    assert!(
        frozen.resolve(&mut renderer).is_err(),
        "a GPU write invalidates the captured content version"
    );
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn graph_normal_frames_draw_directly_and_resolve_msaa_without_a_copy_pass() {
    for quality in [StageQuality::Low, StageQuality::High] {
        let descriptors = gpu();
        let target = TextureTarget::new(&descriptors.device, (32, 32)).unwrap();
        let mut renderer = WgpuRenderBackend::with_options(descriptors, target, false).unwrap();
        renderer.set_quality(quality).unwrap();
        let metrics = ruffle_render::profile::RenderMetrics::default();
        let _scope = ruffle_render::profile::enter(Some(&metrics));
        let mut commands = CommandList::new();
        rect(&mut commands, 0.0, 0.0, 32.0, 32.0, Color::RED);
        renderer
            .submit_frame(Color::BLACK, commands, vec![])
            .unwrap();
        assert_eq!(
            metrics.snapshot().work.render_passes,
            1,
            "normal drawing must use the final attachment"
        );
        assert_eq!(metrics.snapshot().work.draw_calls, 1);
        let image = renderer.capture_frame().unwrap().unwrap();
        assert!(image.pixels().all(|pixel| pixel.0 == [255, 0, 0, 255]));
    }
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn shader_filter_preparation_error_is_returned_without_panicking() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat, BitmapHandle, BitmapHandleImpl};
    use ruffle_render::filters::{Filter, ShaderFilter, ShaderObject};
    use ruffle_render::pixel_bender_support::{ImageInputTexture, PixelBenderShaderArgument};
    #[derive(Debug, Clone)]
    struct TestShaderObject;
    impl ShaderObject for TestShaderObject {
        fn clone_box(&self) -> Box<dyn ShaderObject> {
            Box::new(self.clone())
        }
        fn equals(&self, _: &dyn ShaderObject) -> bool {
            false
        }
    }
    #[derive(Debug)]
    struct ForeignBitmap;
    impl BitmapHandleImpl for ForeignBitmap {}
    let descriptors = gpu();
    let target = TextureTarget::new_without_readback(&descriptors.device, (4, 4)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors, target, false).unwrap();
    let bitmap = renderer
        .register_bitmap(Bitmap::new(4, 4, BitmapFormat::Rgba, vec![255; 64]))
        .unwrap();
    let shader = renderer
        .compile_pixelbender_shader(
            ruffle_render::pixel_bender::parse_shader(
                include_bytes!(
                    "../../../../tests/tests/swfs/visual/blend_modes/shader_as_mask/shader.pbj"
                ),
                false,
            )
            .unwrap(),
        )
        .unwrap();
    let filter = Filter::ShaderFilter(ShaderFilter {
        bottom_extension: 0,
        left_extension: 0,
        right_extension: 0,
        top_extension: 0,
        shader_object: Box::new(TestShaderObject),
        shader,
        shader_args: vec![
            PixelBenderShaderArgument::ImageInput {
                index: 0,
                channels: 4,
                name: "background".into(),
                texture: None,
            },
            PixelBenderShaderArgument::ImageInput {
                index: 1,
                channels: 4,
                name: "foreground".into(),
                texture: Some(ImageInputTexture::Bitmap(BitmapHandle(Arc::new(
                    ForeignBitmap,
                )))),
            },
        ],
    });
    let result = renderer.apply_filter(bitmap.clone(), (0, 0), (4, 4), bitmap, (0, 0), filter);
    assert!(
        matches!(result, Err(ruffle_render::error::Error::Render(error)) if error.kind() == crate::gpu_runtime::RenderFailureKind::Validation)
    );
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn pointwise_filter_chain_preserves_intermediate_rgba8_quantization() {
    let width = 256;
    let height = 16;
    let mut pixels = Vec::new();
    for alpha in [
        0u32, 1, 2, 3, 7, 16, 31, 63, 64, 127, 128, 192, 223, 254, 255, 201,
    ] {
        for x in 0..width {
            pixels.extend([
                (x * alpha / 255) as u8,
                ((255 - x) * alpha / 255) as u8,
                (((x * 17) % 256) * alpha / 255) as u8,
                alpha as u8,
            ]);
        }
    }
    let filters = vec![
        swf::ColorMatrixFilter {
            matrix: [
                1.13, 0.07, -0.1, 0.0, 5.25, 0.0, 0.93, 0.12, 0.0, -3.75, 0.0, -0.13, 1.1, 0.0,
                2.5, 0.0, 0.0, 0.0, 0.8, 3.25,
            ],
        },
        swf::ColorMatrixFilter {
            matrix: [
                0.79, 0.0, 0.0, 0.0, -2.5, 0.0, 1.17, 0.0, 0.0, 1.5, 0.11, 0.0, 0.88, 0.0, 0.5,
                0.0, 0.0, 0.0, 1.1, -1.25,
            ],
        },
        swf::ColorMatrixFilter {
            matrix: [
                1.01, 0.0, 0.0, 0.0, 0.5, 0.0, 0.99, 0.0, 0.0, -0.5, 0.0, 0.0, 1.02, 0.0, 0.25,
                0.0, 0.0, 0.0, 0.75, 0.0,
            ],
        },
    ];
    let results = render_color_matrix_test(&filters, pixels, width, height);
    let mismatch = results[0]
        .0
        .iter()
        .zip(&results[1].0)
        .enumerate()
        .find(|(_, (a, b))| a != b);
    assert!(
        mismatch.is_none(),
        "fusion changed intermediate quantization at {mismatch:?}"
    );
    assert_eq!(results[0].1, 3);
    assert_eq!(
        results[1].1, 3,
        "general matrices must retain native attachment quantization"
    );
}

fn render_color_matrix_test(
    filters: &[swf::ColorMatrixFilter],
    pixels: Vec<u8>,
    width: u32,
    height: u32,
) -> Vec<(Vec<u8>, u64)> {
    let filters: Vec<_> = filters
        .iter()
        .cloned()
        .map(ruffle_render::filters::Filter::ColorMatrixFilter)
        .collect();
    render_filter_chain_test(&filters, pixels, width, height)
}

fn render_filter_chain_test(
    filters: &[ruffle_render::filters::Filter],
    pixels: Vec<u8>,
    width: u32,
    height: u32,
) -> Vec<(Vec<u8>, u64)> {
    use ruffle_render::{
        backend::BitmapCacheEntry,
        bitmap::{Bitmap, BitmapFormat, PixelRegion, PixelSnapping},
        transform::Transform,
    };
    let descriptors = gpu();
    let mut results = Vec::new();
    for chain in [false, true] {
        let target = TextureTarget::new(&descriptors.device, (width, height)).unwrap();
        let mut renderer =
            WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
        renderer.set_quality(StageQuality::Low).unwrap();
        let source = renderer
            .register_bitmap(Bitmap::new(
                width,
                height,
                BitmapFormat::Rgba,
                pixels.clone(),
            ))
            .unwrap();
        let empty = || {
            Bitmap::new(
                width,
                height,
                BitmapFormat::Rgba,
                vec![0; (width * height * 4) as usize],
            )
        };
        let draw = |bitmap| {
            let mut list = CommandList::new();
            list.render_bitmap(
                bitmap,
                Transform::default(),
                false,
                PixelSnapping::Never,
                PixelRegion::for_whole_size(width, height),
            );
            list
        };
        let metrics = ruffle_render::profile::RenderMetrics::default();
        let _scope = ruffle_render::profile::enter(Some(&metrics));
        if chain {
            let destination = renderer.register_bitmap(empty()).unwrap();
            renderer
                .submit_frame(
                    Color::TRANSPARENT,
                    draw(destination.clone()),
                    vec![BitmapCacheEntry {
                        handle: destination,
                        commands: draw(source),
                        clear: Color::TRANSPARENT,
                        filters: filters.to_vec(),
                    }],
                )
                .unwrap();
        } else {
            let mut source = source;
            for filter in filters {
                let destination = renderer.register_bitmap(empty()).unwrap();
                renderer
                    .apply_filter(
                        source,
                        (0, 0),
                        (width, height),
                        destination.clone(),
                        (0, 0),
                        filter.clone(),
                    )
                    .unwrap()
                    .unwrap();
                source = destination;
            }
            renderer
                .submit_frame(Color::TRANSPARENT, draw(source), vec![])
                .unwrap();
        }
        results.push((
            renderer.capture_frame().unwrap().unwrap().into_raw(),
            metrics.snapshot().work.filter_passes,
        ));
    }
    results
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn pointwise_filter_chain_matches_rgba8_rounding_at_half_steps() {
    let filters = [
        swf::ColorMatrixFilter {
            matrix: [
                0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0,
                0.0, 0.0, 0.0, 255.0,
            ],
        },
        swf::ColorMatrixFilter::default(),
    ];
    let results = render_color_matrix_test(&filters, vec![255; 4], 1, 1);
    assert_eq!(results[1].0, results[0].0);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn pointwise_channel_permutations_fuse_without_changing_alpha_or_bytes() {
    let mut rotate = swf::ColorMatrixFilter::default();
    rotate.matrix[0] = 0.0;
    rotate.matrix[1] = 1.0;
    rotate.matrix[6] = 0.0;
    rotate.matrix[7] = 1.0;
    rotate.matrix[12] = 0.0;
    rotate.matrix[10] = 1.0;
    let mut last = swf::ColorMatrixFilter::default();
    last.matrix[0] = 1.13;
    last.matrix[4] = 5.25;
    last.matrix[6] = 0.93;
    last.matrix[9] = -3.75;
    last.matrix[18] = 0.8;
    last.matrix[19] = 3.25;
    let mut pixels = Vec::new();
    for alpha in 0..=255u32 {
        for x in 0..=255u32 {
            pixels.extend([
                (x * alpha / 255) as u8,
                ((255 - x) * alpha / 255) as u8,
                (((x * 17) % 256) * alpha / 255) as u8,
                alpha as u8,
            ]);
        }
    }
    let results = render_color_matrix_test(&[rotate.clone(), rotate, last], pixels, 256, 256);
    let mismatch = results[0]
        .0
        .iter()
        .zip(&results[1].0)
        .enumerate()
        .find(|(_, (a, b))| a != b);
    assert!(
        mismatch.is_none(),
        "channel fusion changed a byte: {mismatch:?}"
    );
    assert_eq!(results[0].1, 3);
    assert_eq!(results[1].1, 1);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn shader_blend_keeps_shader_pixels_inside_a_stencil_mask() {
    for quality in [StageQuality::Low, StageQuality::High] {
        let descriptors = gpu();
        let mut images = Vec::new();
        for masked in [false, true] {
            let target = TextureTarget::new(&descriptors.device, (32, 32)).unwrap();
            let mut renderer =
                WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
            renderer.set_quality(quality).unwrap();
            let shader = renderer.compile_pixelbender_shader(ruffle_render::pixel_bender::parse_shader(include_bytes!("../../../../tests/tests/swfs/visual/blend_modes/shader_as_mask/shader.pbj"), false).unwrap()).unwrap();
            let mut commands = CommandList::new();
            rect(&mut commands, 0.0, 0.0, 32.0, 32.0, Color::BLUE);
            if masked {
                commands.push_mask();
                rect(&mut commands, 0.0, 0.0, 16.0, 32.0, Color::WHITE);
                commands.activate_mask();
            }
            let mut foreground = CommandList::new();
            rect(&mut foreground, 0.0, 0.0, 32.0, 32.0, Color::RED);
            commands.blend(foreground, RenderBlendMode::Shader(shader));
            if masked {
                commands.deactivate_mask();
                rect(&mut commands, 0.0, 0.0, 16.0, 32.0, Color::WHITE);
                commands.pop_mask();
            }
            rect(&mut commands, 0.0, 28.0, 32.0, 4.0, Color::GREEN);
            renderer
                .submit_frame(Color::BLACK, commands, vec![])
                .unwrap();
            images.push(renderer.capture_frame().unwrap().unwrap());
        }
        let expected = images[0].get_pixel(8, 16);
        assert!(
            expected.0[0] > 0 && expected.0[2] > 0,
            "reference must blend both inputs"
        );
        assert_eq!(images[1].get_pixel(8, 16), expected);
        assert_eq!(images[1].get_pixel(24, 16).0, [0, 0, 255, 255]);
        assert_eq!(images[1].get_pixel(8, 30).0, [0, 255, 0, 255]);
        assert_eq!(images[1].get_pixel(24, 30).0, [0, 255, 0, 255]);
    }
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn noop_blur_does_not_allocate_intermediates() {
    let descriptors = gpu();
    let source = descriptors
        .device
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
    let mut pool = crate::buffer_pool::TexturePool::new(crate::idle_pool::Budget::with_runtime(
        &descriptors.runtime,
    ));
    let mut encoder = descriptors
        .device
        .create_command_encoder(&Default::default())
        .unwrap();
    let mut belt = crate::gpu_resources::StagingBelt::new(descriptors.device.clone(), 64);
    let before = descriptors.runtime.stats();
    let result = descriptors
        .filters
        .blur
        .apply(
            &descriptors,
            &mut pool,
            &mut encoder,
            &mut belt,
            &crate::filters::FilterSource::for_entire_texture(&source),
            &swf::BlurFilter {
                blur_x: swf::Fixed16::ONE,
                blur_y: swf::Fixed16::ONE,
                flags: swf::BlurFilterFlags::from_passes(3),
            },
        )
        .unwrap();
    assert!(result.is_none());
    assert_eq!(descriptors.runtime.stats().total_bytes, before.total_bytes);
    assert_eq!(descriptors.runtime.stats().resources, before.resources);
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn filter_chain_preserves_fractional_blur_halos_at_image_edges() {
    use ruffle_render::filters::Filter;
    let filters = [
        Filter::BlurFilter(swf::BlurFilter {
            blur_x: swf::Fixed16::from_f32(5.5),
            blur_y: swf::Fixed16::from_f32(3.5),
            flags: swf::BlurFilterFlags::from_passes(2),
        }),
        Filter::ColorMatrixFilter(Default::default()),
        Filter::BlurFilter(swf::BlurFilter {
            blur_x: swf::Fixed16::from_f32(2.5),
            blur_y: swf::Fixed16::ONE,
            flags: swf::BlurFilterFlags::from_passes(1),
        }),
    ];
    let graph = crate::filters::graph::FilterGraph::new(&filters, true);
    assert_eq!(
        graph.required_input([1, 1, 3, 3], [0, 0, 32, 32]),
        [0, 0, 10, 7]
    );
    let mut pixels = vec![0; 32 * 32 * 4];
    for (x, y) in [(0, 0), (0, 15), (15, 31), (31, 31), (16, 16)] {
        pixels[(y * 32 + x) * 4..(y * 32 + x) * 4 + 4].copy_from_slice(&[255; 4]);
    }
    let results = render_filter_chain_test(&filters, pixels, 32, 32);
    assert_eq!(results[0].0, results[1].0);
    assert!(results[1].0[3] > 0, "edge contribution must survive");
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn adjacent_geometry_merges_only_contiguous_indices_with_identical_instances() {
    use crate::surface::{buffers::DrawOffsets, chunk::DrawCommand, mask::MaskState};
    let descriptors = gpu();
    let offsets = DrawOffsets {
        vertex_buffer: Some(descriptors.quad.vertex_buffer.clone()),
        index_buffer: descriptors.quad.index_buffer.clone(),
        vertices: descriptors.quad.vertices_pos.clone(),
        indices: 0..3,
        instances: 0..1,
    };
    let mut first = DrawCommand::DrawColor {
        draw_offsets: offsets.clone(),
    };
    let mut next = offsets.clone();
    next.indices = 3..6;
    assert!(first.try_merge(&DrawCommand::DrawColor {
        draw_offsets: next.clone()
    }));
    let DrawCommand::DrawColor { draw_offsets } = &first else {
        panic!()
    };
    assert_eq!(draw_offsets.indices, 0..6);
    next.indices = 6..9;
    next.instances = 1..2;
    assert!(!first.try_merge(&DrawCommand::DrawColor {
        draw_offsets: next.clone()
    }));
    next.instances = 0..1;
    next.indices = 9..12;
    assert!(!first.try_merge(&DrawCommand::DrawColor { draw_offsets: next }));
    assert!(!first.try_merge(&DrawCommand::SetMask(MaskState::empty())));
    assert!(!first.try_merge(&DrawCommand::DrawLine {
        draw_offsets: offsets
    }));
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn chained_gpu_copies_preserve_self_overlap_in_both_directions() {
    use ruffle_render::{
        bitmap::{Bitmap, BitmapFormat, PixelRegion, PixelSnapping},
        transform::Transform,
    };
    let descriptors = gpu();
    let target = TextureTarget::new(&descriptors.device, (8, 4)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors, target, false).unwrap();
    renderer.set_quality(StageQuality::Low).unwrap();
    let mut expected = Vec::new();
    for y in 0..4u8 {
        for x in 0..8u8 {
            expected.extend([x * 21, y * 61, (x + y) * 17, 255]);
        }
    }
    let source = renderer
        .register_bitmap(Bitmap::new(8, 4, BitmapFormat::Rgba, expected.clone()))
        .unwrap();
    let destination = renderer
        .register_bitmap(Bitmap::new(8, 4, BitmapFormat::Rgba, vec![0; 128]))
        .unwrap();
    renderer
        .copy_bitmap_region(
            source,
            PixelRegion::for_whole_size(8, 4),
            destination.clone(),
            PixelRegion::for_whole_size(8, 4),
        )
        .unwrap()
        .unwrap();
    for (src, dst) in [
        (
            PixelRegion::for_region(0, 0, 6, 3),
            PixelRegion::for_region(2, 1, 6, 3),
        ),
        (
            PixelRegion::for_region(2, 0, 6, 4),
            PixelRegion::for_region(0, 0, 6, 4),
        ),
    ] {
        let before = expected.clone();
        for y in 0..src.height() {
            for x in 0..src.width() {
                let from = ((src.y_min + y) * 8 + src.x_min + x) as usize * 4;
                let to = ((dst.y_min + y) * 8 + dst.x_min + x) as usize * 4;
                expected[to..to + 4].copy_from_slice(&before[from..from + 4]);
            }
        }
        assert!(
            renderer
                .copy_bitmap_region(destination.clone(), src, destination.clone(), dst)
                .unwrap()
                .is_some(),
            "self-copy must use a budgeted GPU temporary"
        );
    }
    let mut commands = CommandList::new();
    commands.render_bitmap(
        destination,
        Transform::default(),
        false,
        PixelSnapping::Never,
        PixelRegion::for_whole_size(8, 4),
    );
    renderer
        .submit_frame(Color::TRANSPARENT, commands, vec![])
        .unwrap();
    assert_eq!(
        renderer.capture_frame().unwrap().unwrap().into_raw(),
        expected
    );
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn bitmap_region_readback_does_not_capture_unrequested_rows() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat, PixelRegion};
    let descriptors = gpu();
    let target = TextureTarget::new(&descriptors.device, (64, 64)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors, target, false).unwrap();
    let pixels: Vec<u8> = (0..64u8)
        .flat_map(|y| (0..64u8).flat_map(move |x| [x, y, x + y, 255]))
        .collect();
    let bitmap = renderer
        .register_bitmap(Bitmap::new(64, 64, BitmapFormat::Rgba, pixels))
        .unwrap();
    let mut observed = None;
    let result = renderer
        .read_bitmap_region(
            &bitmap,
            PixelRegion::for_pixel(31, 47),
            Box::new(|bytes, stride| {
                observed = Some((bytes.to_vec(), stride));
            }),
        )
        .unwrap();
    assert_eq!(result, Some(()));
    let (bytes, stride) = observed.unwrap();
    assert_eq!(&bytes[..4], &[31, 47, 78, 255]);
    assert_eq!(stride, 256);
    assert_eq!(
        bytes.len(),
        256,
        "readback may pad its row but must not copy unrelated rows"
    );
}
