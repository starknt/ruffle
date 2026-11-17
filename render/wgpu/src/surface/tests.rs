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
    // One shape pass followed by the surface copy pass, no capture pass.
    assert_eq!(work.draw_calls, 2);
    assert_eq!(work.render_passes, 2);
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
