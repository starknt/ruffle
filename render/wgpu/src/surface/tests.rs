//! Opt-in real-device regressions; enable wgpu/counters when running these.
use crate::backend::{WgpuRenderBackend, request_adapter_and_device};
use crate::descriptors::Descriptors;
use crate::target::TextureTarget;
use ruffle_render::backend::RenderBackend;
use ruffle_render::commands::{CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::matrix::Matrix;
use ruffle_render::quality::StageQuality;
use std::sync::Arc;
use swf::{BlendMode, Color, Twips};

fn gpu() -> Arc<Descriptors> {
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
    Arc::new(Descriptors::new(instance, adapter, device, queue))
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
    optimized: bool,
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
    let mut renderer =
        WgpuRenderBackend::with_options(descriptors.clone(), target, optimized, false).unwrap();
    renderer.set_blend_batching(optimized);
    renderer.set_quality(quality);
    let mut frames = Vec::new();
    let mut live = 0;
    for frame in 0..3 {
        renderer.submit_frame(Color::WHITE, scene(frame, masked, parent_alpha), vec![]);
        let image = renderer
            .capture_frame()
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
fn blend_batches_require_explicit_opt_in() {
    let descriptors = gpu();
    for optimized in [false, true] {
        let target = TextureTarget::new(&descriptors.device, (128, 128)).unwrap();
        let mut renderer =
            WgpuRenderBackend::with_options(descriptors.clone(), target, optimized, false).unwrap();
        let metrics = ruffle_render::profile::RenderMetrics::default();
        let _scope = ruffle_render::profile::enter(Some(&metrics), optimized);
        renderer.submit_frame(Color::WHITE, scene(0, true, false), vec![]);
        let expected = renderer.capture_frame().expect("default renderer capture");
        assert_eq!(
            metrics.snapshot().blend_batch_splits,
            0,
            "unvalidated batching must not be enabled by the renderer default"
        );
        for enabled in [true, false, true] {
            renderer.set_blend_batching(enabled);
            // Recreating the viewport pool must preserve the explicit policy.
            renderer.set_viewport_dimensions(renderer.viewport_dimensions());
            let before = metrics.snapshot().blend_batch_splits;
            renderer.submit_frame(Color::WHITE, scene(0, true, false), vec![]);
            let actual = renderer
                .capture_frame()
                .expect("configured renderer capture");
            assert_eq!(actual, expected, "batch policy changed pixels");
            assert_eq!(
                metrics.snapshot().blend_batch_splits > before,
                enabled && optimized,
                "batch policy lost on viewport recreation or applied to stock"
            );
        }
    }
}

#[test]
#[ignore = "requires a real Vulkan adapter and wgpu/counters"]
fn blend_batches_reuse_textures_and_preserve_masked_pixels() {
    let descriptors = gpu();
    for quality in [StageQuality::Low, StageQuality::High] {
        for masked in [false, true] {
            let (stock, stock_live) = render(&descriptors, false, masked, false, quality);
            let (optimized, optimized_live) = render(&descriptors, true, masked, false, quality);
            eprintln!(
                "blendTemporaryBytes quality={quality:?} masked={masked} stock={stock_live} optimized={optimized_live}"
            );
            assert_eq!(
                stock, optimized,
                "blend order, mask state, or reused contents changed: {quality:?}, mask={masked}"
            );
            assert!(
                optimized_live * 4 < stock_live,
                "temporary texture bytes were not bounded: stock={stock_live}, optimized={optimized_live}"
            );
        }
    }
}

#[test]
#[ignore = "requires a real Vulkan adapter and wgpu/counters"]
fn ancestor_alpha_and_erase_preserve_original_layer_sampling() {
    let descriptors = gpu();
    let (stock, _) = render(&descriptors, false, true, true, StageQuality::High);
    let (optimized, _) = render(&descriptors, true, true, true, StageQuality::High);
    assert_eq!(
        stock, optimized,
        "nested Alpha/Erase must sample the original layer contents"
    );
}

#[test]
#[ignore = "optional Vulkan benchmark; emits measurements without a timing threshold"]
fn benchmark_blend_batches_encode_and_submit() {
    let descriptors = gpu();
    for optimized in [false, true] {
        let target = TextureTarget::new(&descriptors.device, (128, 128)).unwrap();
        let mut renderer =
            WgpuRenderBackend::with_options(descriptors.clone(), target, optimized, false).unwrap();
        renderer.set_blend_batching(optimized);
        renderer.set_quality(StageQuality::High);
        let mut timings = Vec::new();
        for frame in 0..34 {
            let commands = scene(frame, true, false);
            let start = std::time::Instant::now();
            renderer.submit_frame(Color::WHITE, commands, vec![]);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            // Keep GPU completion/readback outside the measured host call.
            renderer.capture_frame().expect("benchmark capture failed");
            if frame >= 3 {
                timings.push(elapsed);
            }
        }
        timings.sort_by(f64::total_cmp);
        println!(
            "blendBatchBenchmark optimized={optimized} samples={} encodeSubmitWallP50Ms={:.3} encodeSubmitWallP95Ms={:.3}",
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
    let mut renderer =
        WgpuRenderBackend::with_options(descriptors.clone(), target, true, false).unwrap();
    renderer.set_quality(StageQuality::Low);
    renderer.submit_frame(Color::WHITE, CommandList::new(), vec![]);
    renderer.capture_frame().unwrap();
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
        renderer.submit_frame(Color::WHITE, commands, vec![]);
        let actual = renderer.capture_frame().unwrap();
        assert_eq!(actual.get_pixel(32, 32).0, [255, 0, 0, 255]);
        assert_eq!(actual.get_pixel(96, 32).0, [0, 0, 255, 255]);
        assert_eq!(actual.get_pixel(32, 96).0, [255, 255, 255, 255]);
    }
    // Drain only in this explicit test, never in the eviction implementation.
    renderer.submit_frame(Color::WHITE, CommandList::new(), vec![]);
    renderer.capture_frame().unwrap();
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
fn immutable_bitmap_cache_is_opt_in_and_stock_stays_unmanaged() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat};
    let descriptors = gpu();
    for optimized in [false, true] {
        let target = TextureTarget::new(&descriptors.device, (4, 4)).unwrap();
        let mut renderer =
            WgpuRenderBackend::with_options(descriptors.clone(), target, optimized, false).unwrap();
        let handle = renderer
            .register_bitmap(Bitmap::new(1, 1, BitmapFormat::Rgba, vec![255; 4]))
            .unwrap();
        assert!(!renderer.cache_immutable_bitmap(&handle));
        for limit in [Some(4), Some(0), None] {
            renderer.set_immutable_bitmap_cache_limit(limit);
            assert_eq!(
                renderer.cache_immutable_bitmap(&handle),
                optimized && limit == Some(4)
            );
        }
    }
}
