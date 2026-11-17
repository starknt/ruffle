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
    use crate::target::RenderTargetFrame;
    let first = target.get_next_texture().unwrap().unwrap();
    let second = target.get_next_texture().unwrap().unwrap();
    assert_eq!(
        first.view().lease().id(),
        second.view().lease().id(),
        "unchanged target reuses its native view"
    );
    target
        .resize(&descriptors.device, 64, 48)
        .expect("GPU operation failed");
    assert!(target.buffer.is_none());
    assert_eq!((target.width(), target.height()), (64, 48));
    let resized = target.get_next_texture().unwrap().unwrap();
    assert_ne!(resized.view().lease().id(), first.view().lease().id());
    assert_eq!(first.view().texture_handle().unwrap().width(), 32);
    assert_eq!(resized.view().texture_handle().unwrap().width(), 64);
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
    let snapshot = metrics.snapshot();
    let resources = snapshot
        .native_resources
        .as_ref()
        .expect("native resources published");
    assert!(resources.logical_bytes > 0);
    assert_eq!(resources.worker_active, 0);
    assert!(
        snapshot.native_timings[ruffle_render::profile::NativePhase::Preparation as usize].samples
            > 0
    );
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
#[ignore = "requires a real native adapter"]
fn adjacent_bitmap_textures_batch_without_reordering_translucent_pixels() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat, PixelRegion, PixelSnapping};
    use ruffle_render::transform::Transform;
    for descriptors in [gpu(), gpu_with_backend(wgpu::Backends::GL, None, None)] {
        for count in [32, 37] {
            for quality in [StageQuality::Low, StageQuality::High] {
                for smoothing in [false, true] {
                    let mut results = Vec::new();
                    for reference in [true, false] {
                        let target = TextureTarget::new(&descriptors.device, (64, 40)).unwrap();
                        let mut renderer =
                            WgpuRenderBackend::with_options(descriptors.clone(), target, false)
                                .unwrap();
                        renderer.set_quality(quality).unwrap();
                        let mut commands = CommandList::new();
                        // The first bitmap's instance index deliberately starts above zero.
                        rect(&mut commands, 0.0, 0.0, 1.0, 1.0, Color::TRANSPARENT);
                        for i in 0..count {
                            if count == 37 && i == 7 {
                                commands.push_mask();
                                rect(&mut commands, 3.5, 2.25, 23.0, 19.0, Color::WHITE);
                                commands.activate_mask();
                            } else if count == 37 && i == 26 {
                                commands.deactivate_mask();
                                rect(&mut commands, 3.5, 2.25, 23.0, 19.0, Color::WHITE);
                                commands.pop_mask();
                            }
                            let alpha = 128 + i as u8;
                            let bitmap = renderer
                                .register_bitmap(Bitmap::new(
                                    2,
                                    2,
                                    BitmapFormat::Rgba,
                                    vec![
                                        0,
                                        0,
                                        0,
                                        0,
                                        alpha / 2,
                                        i as u8 * 2,
                                        alpha,
                                        alpha,
                                        1,
                                        0,
                                        1,
                                        1,
                                        255,
                                        128,
                                        0,
                                        255,
                                    ],
                                ))
                                .unwrap();
                            commands.render_bitmap(
                                bitmap,
                                Transform {
                                    matrix: Matrix {
                                        a: 3.5,
                                        d: 3.0,
                                        tx: Twips::from_pixels(1.25 + f64::from(i % 8) * 4.0),
                                        ty: Twips::from_pixels(2.5 + f64::from(i / 8) * 4.0),
                                        ..Matrix::IDENTITY
                                    },
                                    color_transform: swf::ColorTransform {
                                        r_multiply: swf::Fixed8::from_f32(1.5),
                                        g_multiply: swf::Fixed8::from_f32(0.75),
                                        a_multiply: swf::Fixed8::from_f32(0.8),
                                        r_add: -30,
                                        b_add: 80,
                                        a_add: i as i16 - 20,
                                        ..swf::ColorTransform::IDENTITY
                                    },
                                    ..Default::default()
                                },
                                smoothing ^ (count == 37 && i >= 17),
                                PixelSnapping::Never,
                                PixelRegion::for_whole_size(2, 2),
                            );
                            if reference {
                                rect(&mut commands, 0.0, 0.0, 1.0, 1.0, Color::TRANSPARENT);
                            }
                        }
                        let metrics = ruffle_render::profile::RenderMetrics::default();
                        let _scope = ruffle_render::profile::enter(Some(&metrics));
                        renderer
                            .submit_frame(Color::WHITE, commands, vec![])
                            .unwrap();
                        if !reference && count == 32 {
                            assert_eq!(
                                metrics.snapshot().work.draw_calls,
                                3,
                                "two bounded bitmap batches and one transparent rectangle"
                            );
                        }
                        let image = renderer.capture_frame().unwrap().unwrap();
                        assert_eq!(image.get_pixel(63, 39).0, [255, 255, 255, 255]);
                        results.push(image);
                    }
                    assert_eq!(
                        results[0], results[1],
                        "{quality:?}, smoothing={smoothing}, count={count}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a real native adapter"]
fn bitmap_batch_bindings_reuse_only_while_every_source_image_is_alive() {
    use ruffle_render::bitmap::{PixelRegion, PixelSnapping};
    use ruffle_render::transform::Transform;
    let descriptors = gpu();
    let target = TextureTarget::new(&descriptors.device, (32, 32)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
    let mut bitmaps: Vec<_> = (0..32)
        .map(|_| {
            renderer
                .create_empty_texture(
                    std::num::NonZeroU32::new(2).unwrap(),
                    std::num::NonZeroU32::new(2).unwrap(),
                )
                .unwrap()
        })
        .collect();
    let groups = || {
        descriptors
            .runtime
            .resources()
            .into_iter()
            .filter(|r| {
                r.kind == crate::gpu_runtime::ResourceKind::BindGroup
                    && r.dependencies
                        .contains(&descriptors.bind_layouts.bitmap_batch.lease().id())
            })
            .map(|r| r.id)
            .collect::<std::collections::HashSet<_>>()
    };
    let mut previous = None;
    for _ in 0..6 {
        let mut commands = CommandList::new();
        for bitmap in &bitmaps {
            commands.render_bitmap(
                bitmap.clone(),
                Transform::default(),
                false,
                PixelSnapping::Never,
                PixelRegion::for_whole_size(2, 2),
            );
        }
        renderer
            .submit_frame(Color::WHITE, commands, vec![])
            .unwrap();
        renderer.capture_frame().unwrap().unwrap();
        descriptors
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let live = groups();
        assert_eq!(live.len(), 2, "reusable bindings outlive one submission");
        if let Some(previous) = previous {
            assert_eq!(
                live, previous,
                "stable images must reuse the same native bindings"
            );
        }
        previous = Some(live);
    }
    // Dropping any source invalidates its shared binding even when the other
    // source images and the descriptor cache stay alive.
    bitmaps.pop();
    assert_eq!(groups().len(), 1);
    drop(bitmaps);
    assert!(groups().is_empty());
}

#[test]
#[ignore = "requires a real native adapter"]
fn alpha_mask_samples_aligned_bitmap_content_without_an_intermediate_copy() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat, PixelRegion, PixelSnapping};
    use ruffle_render::transform::Transform;
    for descriptors in [gpu(), gpu_with_backend(wgpu::Backends::GL, None, None)] {
        for (quality, smoothing) in [StageQuality::Low, StageQuality::High]
            .into_iter()
            .flat_map(|q| [false, true].map(|s| (q, s)))
        {
            for (x, scale) in [(-3.0, 1.0), (7.0, 1.0), (7.25, 1.0), (7.0, 1.5)] {
                let mut images = Vec::new();
                for reference in [true, false] {
                    let target = TextureTarget::new(&descriptors.device, (32, 24)).unwrap();
                    let mut renderer =
                        WgpuRenderBackend::with_options(descriptors.clone(), target, false)
                            .unwrap();
                    renderer.set_quality(quality).unwrap();
                    let pixels: Vec<u8> = (0..48)
                        .flat_map(|i| {
                            let a = [0, 1, 127, 255][i % 4];
                            [a / 2, a / 3, a, a]
                        })
                        .collect();
                    let bitmap = renderer
                        .register_bitmap(Bitmap::new(8, 6, BitmapFormat::Rgba, pixels))
                        .unwrap();
                    let mut content = CommandList::new();
                    if reference {
                        // A transparent draw preserves pixels while requiring general composition.
                        rect(
                            &mut content,
                            x as f32,
                            4.0,
                            8.0 * scale,
                            6.0 * scale,
                            Color::TRANSPARENT,
                        );
                    }
                    content.render_bitmap(
                        bitmap,
                        Transform {
                            matrix: Matrix {
                                a: scale,
                                d: scale,
                                tx: Twips::from_pixels(x),
                                ty: Twips::from_pixels(4.0),
                                ..Matrix::IDENTITY
                            },
                            ..Default::default()
                        },
                        smoothing,
                        PixelSnapping::Never,
                        PixelRegion::for_whole_size(8, 6),
                    );
                    let mut mask = CommandList::new();
                    rect(
                        &mut mask,
                        x as f32 + 1.0,
                        5.0,
                        7.0,
                        7.0,
                        Color::from_rgb(0xffffff, 137),
                    );
                    let mut commands = CommandList::new();
                    commands.render_alpha_mask(content, mask);
                    let nested = x == 7.0 && scale == 1.0;
                    if nested {
                        let mut parent = CommandList::new();
                        parent.blend(commands, RenderBlendMode::Builtin(BlendMode::Layer));
                        commands = parent;
                    }
                    let metrics = ruffle_render::profile::RenderMetrics::default();
                    let _scope = ruffle_render::profile::enter(Some(&metrics));
                    renderer
                        .submit_frame(Color::WHITE, commands, vec![])
                        .unwrap();
                    let eligible = !reference && x.fract() == 0.0 && scale == 1.0;
                    assert_eq!(
                        metrics.snapshot().work.offscreen_targets,
                        if eligible { 1 } else { 2 } + u64::from(nested)
                    );
                    let image = renderer.capture_frame().unwrap().unwrap();
                    assert_eq!(image.get_pixel(31, 23).0, [255, 255, 255, 255]);
                    images.push(image);
                }
                assert_eq!(
                    images[0], images[1],
                    "quality={quality:?}, x={x}, scale={scale}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a real native adapter"]
fn chunk_uniform_windows_reuse_the_same_buffer_binding() {
    let descriptors = gpu();
    let mut buffers = super::buffers::ChunkBuffers::new(&descriptors).unwrap();
    let mut staging = crate::gpu_resources::StagingBelt::new(descriptors.device.clone(), 65536);
    let mut encoder = descriptors
        .device
        .create_command_encoder(&Default::default())
        .unwrap();
    let mut bindings = Vec::new();
    for x in [3.0, 19.0, 7.0] {
        let mut chunk = super::buffers::ChunkBufferBuilder::new(&buffers, &descriptors.limits);
        chunk
            .add_transform(&ruffle_render::transform::Transform {
                matrix: Matrix::translate(Twips::from_pixels(x), Twips::ZERO),
                ..Default::default()
            })
            .unwrap();
        chunk
            .copy_to(&mut staging, &descriptors, &mut encoder, &mut buffers)
            .unwrap();
        bindings.push(buffers.bind_group.lease().id());
    }
    assert!(
        bindings.iter().all(|id| *id == bindings[0]),
        "one arena page needs one transform binding, independent of slice offsets: {bindings:?}"
    );
    staging.finish();
    descriptors
        .queue
        .submit([encoder.finish().unwrap()])
        .unwrap();
    descriptors
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

#[test]
#[ignore = "requires a real native adapter"]
fn builtin_quad_draws_do_not_upload_immutable_vertices_again() {
    use super::buffers::{ChunkBufferBuilder, ChunkBuffers, DrawVertices};
    let descriptors = gpu();
    let mut buffers = ChunkBuffers::new(&descriptors).unwrap();
    let original_dynamic = buffers.vertex_buffer.lease().id();
    let mut chunk = ChunkBufferBuilder::new(&buffers, &descriptors.limits);
    let instance = chunk.add_transform(&Default::default()).unwrap();
    for vertices in [
        DrawVertices::PosUvQuad,
        DrawVertices::PosColorQuad,
        DrawVertices::PosQuad,
        DrawVertices::Line,
        DrawVertices::RectLine,
    ] {
        let draw = chunk
            .add_draw(&descriptors.quad, instance, vertices)
            .unwrap();
        assert_eq!(
            draw.vertex_buffer.as_ref().map(|b| b.lease().id()),
            Some(descriptors.quad.vertex_buffer.lease().id()),
            "built-in geometry already has immutable native storage"
        );
    }
    let mut staging = crate::gpu_resources::StagingBelt::new(descriptors.device.clone(), 65536);
    let mut encoder = descriptors
        .device
        .create_command_encoder(&Default::default())
        .unwrap();
    chunk
        .copy_to(&mut staging, &descriptors, &mut encoder, &mut buffers)
        .unwrap();
    assert_eq!(
        buffers.vertex_buffer.lease().id(),
        original_dynamic,
        "no custom vertices need a dynamic upload"
    );
    staging.finish();
    descriptors
        .queue
        .submit([encoder.finish().unwrap()])
        .unwrap();
    descriptors
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

#[test]
#[ignore = "requires a real native adapter"]
fn small_alpha_masks_share_a_pass_within_the_transient_byte_budget() {
    let split = gpu_with_watermark(Some(2048));
    let combined = gpu_with_watermark(Some(1024 * 1024));
    for quality in [StageQuality::Low, StageQuality::High] {
        let mut images = Vec::new();
        for (index, descriptors) in [&split, &combined].into_iter().enumerate() {
            let target = TextureTarget::new(&descriptors.device, (96, 64)).unwrap();
            let mut renderer =
                WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
            renderer.set_quality(quality).unwrap();
            let metrics = ruffle_render::profile::RenderMetrics::default();
            let _scope = ruffle_render::profile::enter(Some(&metrics));
            let mut commands = CommandList::new();
            for i in 0..24 {
                let x = 3.25 + (i % 6) as f32 * 9.0;
                let y = 4.5 + (i / 6) as f32 * 9.0;
                let mut content = CommandList::new();
                rect(
                    &mut content,
                    x,
                    y,
                    12.5,
                    12.0,
                    Color::from_rgb(if i % 2 == 0 { 0xc35b28 } else { 0x3583dc }, 179),
                );
                let mut mask = CommandList::new();
                rect(
                    &mut mask,
                    x + 2.5,
                    y + 1.25,
                    10.0,
                    10.5,
                    Color::from_rgb(0xffffff, 127),
                );
                commands.render_alpha_mask(content, mask);
            }
            renderer
                .submit_frame(Color::WHITE, commands, vec![])
                .unwrap();
            let snapshot = metrics.snapshot();
            if index == 0 {
                assert!(
                    snapshot.blend_batch_splits > 0,
                    "byte budget must split masks"
                );
            } else {
                assert_eq!(
                    snapshot.work.mask_passes, 1,
                    "small inputs fit one composite pass"
                );
                assert_eq!(snapshot.work.render_passes, 49);
                assert_eq!(snapshot.blend_batch_splits, 0);
            }
            let image = renderer.capture_frame().unwrap().unwrap();
            assert_eq!(image.get_pixel(95, 63).0, [255, 255, 255, 255]);
            assert_ne!(image.get_pixel(8, 8).0, [255, 255, 255, 255]);
            images.push(image);
        }
        assert_eq!(images[0], images[1], "quality {quality:?}");
    }
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
    gpu_with_budgets(watermark, None)
}

fn gpu_with_budgets(watermark: Option<u64>, staging: Option<u64>) -> Arc<Descriptors> {
    gpu_with_backend(wgpu::Backends::VULKAN, watermark, staging)
}

fn gpu_with_backend(
    backends: wgpu::Backends,
    watermark: Option<u64>,
    staging: Option<u64>,
) -> Arc<Descriptors> {
    let instance = crate::backend::create_wgpu_instance(backends, Default::default(), None);
    let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
        backends,
        &instance,
        None,
        wgpu::PowerPreference::HighPerformance,
    ))
    .expect("explicit GPU regressions require the requested native adapter");
    eprintln!("native GPU regression adapter: {:?}", adapter.get_info());
    let mut config =
        crate::gpu_runtime::GpuBudgetConfig::for_adapter(adapter.get_info().device_type, false);
    if let Some(watermark) = watermark {
        config.transient_watermark = watermark;
    }
    if let Some(staging) = staging {
        config.staging_limit = staging;
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
fn async_readback_completion_and_cancellation_return_unmapped_pool_entries() {
    let descriptors = gpu();
    let target = TextureTarget::new_without_readback(&descriptors.device, (1, 1)).unwrap();
    let renderer = WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
    let dimensions = crate::utils::BufferDimensions::new(1, 1, wgpu::TextureFormat::Rgba8Unorm);
    for complete in [true, false, true] {
        let buffer = renderer
            .offscreen_buffer_pool
            .take(&descriptors, dimensions.clone())
            .unwrap();
        let mut readback =
            crate::AsyncReadback::new(buffer, dimensions.clone(), descriptors.clone(), None);
        if complete {
            descriptors
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let (pixels, stride) = readback.poll().unwrap().unwrap();
            assert_eq!(stride, 256);
            assert!(pixels.iter().all(|&byte| byte == 0));
        }
        drop(readback);
        descriptors
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        descriptors.runtime.check_health().unwrap();
    }
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
    commands.render_shape(handle.clone(), Transform::default());
    renderer
        .submit_frame(Color::BLACK, commands, vec![])
        .unwrap();
    let recipe = (&*handle.0 as &dyn std::any::Any)
        .downcast_ref::<crate::shape_recipe::ShapeRecipe>()
        .unwrap();
    let buffers: Vec<_> = descriptors
        .runtime
        .resources()
        .into_iter()
        .filter(|r| r.owner == recipe.owner && r.kind == crate::gpu_runtime::ResourceKind::Buffer)
        .collect();
    assert_eq!(
        buffers.len(),
        1,
        "one immutable mesh fits in one native allocation"
    );
    let image = renderer.capture_frame().unwrap().unwrap();
    assert!(image.pixels().all(|p| p.0 == [255, 0, 0, 255]));
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn equal_small_meshes_batch_without_retaining_unloaded_shape_owners() {
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
    let target = TextureTarget::new(&descriptors.device, (24, 8)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
    let mut handles = Vec::new();
    for (id, color) in [(1, Color::RED), (2, Color::RED), (3, Color::BLUE)] {
        let style = swf::FillStyle::Color(color);
        let point = |x, y| swf::Point::new(Twips::from_pixels(x), Twips::from_pixels(y));
        handles.push(
            renderer
                .register_shape(
                    DistilledShape {
                        id,
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
                .unwrap(),
        );
    }
    let metrics = ruffle_render::profile::RenderMetrics::default();
    let _scope = ruffle_render::profile::enter(Some(&metrics));
    let mut commands = CommandList::new();
    for (i, handle) in handles.iter().enumerate() {
        let mut transform = Transform::default();
        transform.matrix.tx = Twips::from_pixels(i as f64 * 8.0);
        commands.render_shape(handle.clone(), transform);
    }
    renderer
        .submit_frame(Color::BLACK, commands, vec![])
        .unwrap();
    assert_eq!(
        metrics.snapshot().work.draw_calls,
        2,
        "equal immutable geometry must form one instance batch"
    );
    let image = renderer.capture_frame().unwrap().unwrap();
    for (x, _, pixel) in image.enumerate_pixels() {
        assert_eq!(
            pixel.0,
            if x < 16 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        );
    }
    let buffers: std::collections::HashSet<_> = handles
        .iter()
        .map(|h| crate::mesh::as_mesh(h).unwrap().vertex_buffer.lease().id())
        .collect();
    assert_eq!(buffers.len(), 2);
    drop(handles);
    // Keep the renderer/cache alive. It must not keep any shape's GPU allocation.
    renderer
        .submit_frame(Color::BLACK, CommandList::new(), vec![])
        .unwrap();
    descriptors
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(2)),
        })
        .unwrap();
    descriptors.runtime.purge_idle_and_cached_resources();
    assert!(
        descriptors
            .runtime
            .resources()
            .iter()
            .all(|r| !buffers.contains(&r.id))
    );
}

#[test]
#[ignore = "requires a real native adapter"]
fn repeated_solid_blend_inputs_share_rasterization_without_changing_pixels() {
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
    for descriptors in [gpu(), gpu_with_backend(wgpu::Backends::GL, None, None)] {
        for quality in [StageQuality::Low, StageQuality::High] {
            for (fractional, variant) in [(0.0, 0), (0.25, 0), (0.25, 1), (0.25, 2)] {
                let mut images = Vec::new();
                for reference in [true, false] {
                    let target = TextureTarget::new(&descriptors.device, (128, 64)).unwrap();
                    let mut renderer =
                        WgpuRenderBackend::with_options(descriptors.clone(), target, false)
                            .unwrap();
                    renderer.set_quality(quality).unwrap();
                    let style = swf::FillStyle::Color(Color::from_rgb(0x48a0dd, 179));
                    let point =
                        |x, y| swf::Point::new(Twips::from_pixels(x), Twips::from_pixels(y));
                    let shape = renderer
                        .register_shape(
                            DistilledShape {
                                id: 1,
                                shape_bounds: Default::default(),
                                edge_bounds: Default::default(),
                                paths: vec![DrawPath::Fill {
                                    style: &style,
                                    winding_rule: FillRule::EvenOdd,
                                    commands: vec![
                                        DrawCommand::MoveTo(point(0.0, 0.0)),
                                        DrawCommand::LineTo(point(12.0, 0.0)),
                                        DrawCommand::LineTo(point(12.0, 12.0)),
                                        DrawCommand::LineTo(point(0.0, 12.0)),
                                        DrawCommand::LineTo(point(0.0, 0.0)),
                                    ],
                                }],
                            },
                            &NoBitmaps,
                        )
                        .unwrap();
                    // Prepare the shape's exact mesh bounds before measuring reuse.
                    let mut warmup = CommandList::new();
                    warmup.render_shape(shape.clone(), Transform::default());
                    renderer.submit_frame(Color::WHITE, warmup, vec![]).unwrap();
                    let metrics = ruffle_render::profile::RenderMetrics::default();
                    let _scope = ruffle_render::profile::enter(Some(&metrics));
                    let mut commands = CommandList::new();
                    for i in 0..24 {
                        let x = 4.0 + (i % 8) as f64 * 10.0 + fractional;
                        let y = 4.0 + (i / 8) as f64 * 10.0 + fractional;
                        let mut child = CommandList::new();
                        let mut transform = Transform {
                            matrix: Matrix::translate(Twips::from_pixels(x), Twips::from_pixels(y)),
                            ..Default::default()
                        };
                        if i % 2 == 0 {
                            if variant == 1 {
                                transform.color_transform.b_add = -80;
                            }
                            if variant == 2 {
                                transform.matrix.a = 1.5;
                            }
                        }
                        child.render_shape(shape.clone(), transform);
                        if reference {
                            rect(
                                &mut child,
                                x as f32,
                                y as f32,
                                12.0,
                                12.0,
                                Color::TRANSPARENT,
                            );
                        }
                        commands.blend(child, RenderBlendMode::Builtin(BlendMode::Multiply));
                    }
                    renderer
                        .submit_frame(Color::WHITE, commands, vec![])
                        .unwrap();
                    if !reference {
                        assert_eq!(
                            metrics.snapshot().work.offscreen_targets,
                            if variant == 0 { 1 } else { 2 },
                            "identical solid input is rasterized once per bounded sibling set"
                        );
                    }
                    let image = renderer.capture_frame().unwrap().unwrap();
                    assert_eq!(image.get_pixel(127, 63).0, [255, 255, 255, 255]);
                    images.push(image);
                }
                assert_eq!(
                    images[0], images[1],
                    "{quality:?}, fractional={fractional}, variant={variant}"
                );
            }
        }
    }
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
    assert!(
        frozen
            .prepare(
                &mut renderer,
                ruffle_render::region::RenderRect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4
                },
                1,
                crate::pass_graph::Load::Load
            )
            .is_err()
    );
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
        frozen
            .prepare(
                &mut renderer,
                ruffle_render::region::RenderRect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4
                },
                1,
                crate::pass_graph::Load::Load
            )
            .is_err(),
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
fn compositing_uses_copyable_final_targets_without_a_full_surface_copy() {
    let descriptors = gpu();
    for quality in [StageQuality::Low, StageQuality::High] {
        for alpha_mask in [false, true] {
            for copyable in [false, true] {
                let mut target = TextureTarget::new(&descriptors.device, (32, 32)).unwrap();
                if !copyable {
                    // A presentation surface may support attachment writes only.
                    target.texture = descriptors
                        .device
                        .create_texture(&wgpu::TextureDescriptor {
                            label: None,
                            size: target.size,
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: target.format,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                            view_formats: &[],
                        })
                        .unwrap();
                    target.buffer = None;
                }
                let mut renderer =
                    WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
                renderer.set_quality(quality).unwrap();
                let metrics = ruffle_render::profile::RenderMetrics::default();
                let _scope = ruffle_render::profile::enter(Some(&metrics));
                let mut foreground = CommandList::new();
                rect(&mut foreground, 8.0, 8.0, 16.0, 16.0, Color::RED);
                let mut commands = CommandList::new();
                if alpha_mask {
                    let mut mask = CommandList::new();
                    rect(&mut mask, 8.0, 8.0, 16.0, 16.0, Color::WHITE);
                    commands.render_alpha_mask(foreground, mask);
                } else {
                    commands.blend(foreground, RenderBlendMode::Builtin(BlendMode::Multiply));
                }
                renderer
                    .submit_frame(Color::WHITE, commands, vec![])
                    .unwrap();
                // Source/mask/composite, or source/clear/blend. An attachment-only
                // surface additionally needs the final copy; a readable target does not.
                assert_eq!(
                    metrics.snapshot().work.render_passes,
                    if copyable { 3 } else { 4 }
                );
                descriptors
                    .device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(std::time::Duration::from_secs(2)),
                    })
                    .unwrap();
                descriptors.runtime.check_health().unwrap();
                if copyable {
                    let image = renderer.capture_frame().unwrap().unwrap();
                    for (x, y, pixel) in image.enumerate_pixels() {
                        let expected = if (8..24).contains(&x) && (8..24).contains(&y) {
                            [255, 0, 0, 255]
                        } else {
                            [255, 255, 255, 255]
                        };
                        assert_eq!(
                            pixel.0, expected,
                            "{quality:?}, alpha={alpha_mask}, ({x},{y})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn resolved_compositing_preserves_msaa_samples_outside_modified_regions() {
    let descriptors = gpu();
    for quality in [StageQuality::Low, StageQuality::High] {
        for overlap in [false, true] {
            let mut images = Vec::new();
            for stencil_reference in [true, false] {
                let target = TextureTarget::new(&descriptors.device, (96, 48)).unwrap();
                let mut renderer =
                    WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
                renderer.set_quality(quality).unwrap();
                let mut commands = CommandList::new();
                if stencil_reference {
                    commands.push_mask();
                    rect(&mut commands, 0.0, 0.0, 96.0, 48.0, Color::WHITE);
                    commands.activate_mask();
                }
                // Partially covered samples outside the composite regions must
                // survive until the final translucent geometry is drawn.
                rect(
                    &mut commands,
                    0.25,
                    1.25,
                    94.0,
                    39.5,
                    Color::from_rgb(0x30b040, 144),
                );
                for (i, mode) in [BlendMode::Multiply, BlendMode::Lighten, BlendMode::Overlay]
                    .into_iter()
                    .enumerate()
                {
                    let x = if overlap && i == 1 {
                        12.0
                    } else {
                        4.0 + i as f32 * 28.0
                    };
                    let mut child = CommandList::new();
                    rect(
                        &mut child,
                        x + 0.25,
                        5.5,
                        12.0,
                        24.0,
                        Color::from_rgb(0x4070c0 + i as u32 * 0x302010, 160),
                    );
                    commands.blend(child, RenderBlendMode::Builtin(mode));
                }
                rect(
                    &mut commands,
                    0.5,
                    1.75,
                    94.0,
                    39.25,
                    Color::from_rgb(0xe04090, 128),
                );
                if stencil_reference {
                    commands.deactivate_mask();
                    rect(&mut commands, 0.0, 0.0, 96.0, 48.0, Color::WHITE);
                    commands.pop_mask();
                }
                renderer
                    .submit_frame(Color::from_rgb(0xc09060, 255), commands, vec![])
                    .unwrap();
                images.push(renderer.capture_frame().unwrap().unwrap().into_raw());
            }
            assert_eq!(images[0], images[1], "{quality:?}, overlap={overlap}");
        }
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

#[test]
#[ignore = "requires a Vulkan adapter"]
fn bitmap_cache_partial_redraw_clears_old_pixels_and_replays_all_translucent_contributors() {
    use ruffle_render::{
        backend::BitmapCacheEntry,
        bitmap::{PixelRegion, PixelSnapping},
    };
    let descriptors = gpu();
    for quality in [StageQuality::Low, StageQuality::High] {
        let mut renderer = WgpuRenderBackend::with_options(
            descriptors.clone(),
            TextureTarget::new(&descriptors.device, (128, 128)).unwrap(),
            false,
        )
        .unwrap();
        renderer.set_quality(quality).unwrap();
        let cache = renderer
            .create_bitmap_cache(128.try_into().unwrap(), 128.try_into().unwrap())
            .unwrap();
        let contents = |position: Option<f32>| {
            let mut commands = CommandList::new();
            rect(
                &mut commands,
                0.0,
                0.0,
                128.0,
                128.0,
                Color {
                    r: 20,
                    g: 40,
                    b: 80,
                    a: 255,
                },
            );
            if let Some(x) = position {
                rect(
                    &mut commands,
                    x,
                    8.25,
                    14.5,
                    12.5,
                    Color {
                        r: 200,
                        g: 0,
                        b: 0,
                        a: 120,
                    },
                );
            }
            rect(
                &mut commands,
                12.25,
                10.5,
                16.25,
                15.25,
                Color {
                    r: 0,
                    g: 200,
                    b: 0,
                    a: 100,
                },
            );
            commands
        };
        for (index, position) in [Some(6.25), Some(18.75), None].into_iter().enumerate() {
            let metrics = ruffle_render::profile::RenderMetrics::default();
            let pixels = {
                let _scope = ruffle_render::profile::enter(Some(&metrics));
                let mut stage = CommandList::new();
                stage.render_bitmap(
                    cache.clone(),
                    Default::default(),
                    false,
                    PixelSnapping::Never,
                    PixelRegion::for_whole_size(128, 128),
                );
                renderer
                    .submit_frame(
                        Color::BLACK,
                        stage,
                        vec![BitmapCacheEntry {
                            handle: cache.clone(),
                            commands: contents(position),
                            clear: Color::TRANSPARENT,
                            filters: vec![],
                        }],
                    )
                    .unwrap();
                renderer.capture_frame().unwrap().unwrap().into_raw()
            };
            let mut reference = WgpuRenderBackend::with_options(
                descriptors.clone(),
                TextureTarget::new(&descriptors.device, (128, 128)).unwrap(),
                false,
            )
            .unwrap();
            reference.set_quality(quality).unwrap();
            let fresh = reference
                .create_bitmap_cache(128.try_into().unwrap(), 128.try_into().unwrap())
                .unwrap();
            let mut stage = CommandList::new();
            stage.render_bitmap(
                fresh.clone(),
                Default::default(),
                false,
                PixelSnapping::Never,
                PixelRegion::for_whole_size(128, 128),
            );
            reference
                .submit_frame(
                    Color::BLACK,
                    stage,
                    vec![BitmapCacheEntry {
                        handle: fresh,
                        commands: contents(position),
                        clear: Color::TRANSPARENT,
                        filters: vec![],
                    }],
                )
                .unwrap();
            assert_eq!(
                pixels,
                reference.capture_frame().unwrap().unwrap().into_raw(),
                "quality {quality:?}, update {index}"
            );
            if index > 0 {
                assert!(
                    metrics.snapshot().work.offscreen_pixels < 128 * 128,
                    "dirty redraw still rasterized the entire cache"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn bitmap_cache_retention_is_reclaimable_without_invalidating_recorded_commands() {
    use ruffle_render::bitmap::{PixelRegion, PixelSnapping};
    let descriptors = gpu();
    let mut renderer = WgpuRenderBackend::with_options(
        descriptors.clone(),
        TextureTarget::new(&descriptors.device, (32, 32)).unwrap(),
        false,
    )
    .unwrap();
    let bitmap = renderer
        .create_bitmap_cache(32.try_into().unwrap(), 32.try_into().unwrap())
        .unwrap();
    let lease = renderer.retain_bitmap_cache(&bitmap).unwrap();
    let weak = Arc::downgrade(&bitmap.0);
    let mut stage = CommandList::new();
    stage.render_bitmap(
        bitmap,
        Default::default(),
        false,
        PixelSnapping::Never,
        PixelRegion::for_whole_size(32, 32),
    );
    assert!(lease.bitmap().is_some());
    descriptors.runtime.scheduler().reclaim();
    assert!(
        lease.bitmap().is_none(),
        "governor did not reclaim cache ownership"
    );
    assert!(
        weak.upgrade().is_some(),
        "already-recorded draw lost its resource"
    );
    renderer.submit_frame(Color::BLACK, stage, vec![]).unwrap();
    renderer.capture_frame().unwrap();
    assert!(weak.upgrade().is_none());

    let bitmap = renderer
        .create_bitmap_cache(16.try_into().unwrap(), 16.try_into().unwrap())
        .unwrap();
    let lease = renderer.retain_bitmap_cache(&bitmap).unwrap();
    let weak = Arc::downgrade(&bitmap.0);
    drop(bitmap);
    drop(lease);
    assert!(
        weak.upgrade().is_none(),
        "unloaded display owner retained a cache"
    );
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn bitmap_cache_filtered_dirty_halo_matches_full_redraw_at_edges_and_with_msaa() {
    use ruffle_render::{
        backend::BitmapCacheEntry,
        bitmap::{PixelRegion, PixelSnapping},
        filters::Filter,
    };
    let descriptors = gpu();
    let blur = Filter::BlurFilter(swf::BlurFilter {
        blur_x: swf::Fixed16::from_f32(5.25),
        blur_y: swf::Fixed16::from_f32(3.75),
        flags: swf::BlurFilterFlags::from_passes(2),
    });
    let glow = Filter::GlowFilter(swf::GlowFilter {
        color: Color {
            r: 50,
            g: 120,
            b: 255,
            a: 170,
        },
        blur_x: swf::Fixed16::from_f32(7.25),
        blur_y: swf::Fixed16::from_f32(5.5),
        strength: swf::Fixed8::from_f32(1.3),
        flags: swf::GlowFilterFlags::from_passes(2) | swf::GlowFilterFlags::COMPOSITE_SOURCE,
    });
    let shadow = Filter::DropShadowFilter(swf::DropShadowFilter {
        color: Color {
            r: 30,
            g: 90,
            b: 120,
            a: 170,
        },
        blur_x: swf::Fixed16::from_f32(5.25),
        blur_y: swf::Fixed16::from_f32(3.75),
        angle: swf::Fixed16::from_f32(0.6),
        distance: swf::Fixed16::from_f32(4.25),
        strength: swf::Fixed8::ONE,
        flags: swf::DropShadowFilterFlags::from_passes(2)
            | swf::DropShadowFilterFlags::COMPOSITE_SOURCE,
    });
    let bevel = Filter::BevelFilter(swf::BevelFilter {
        shadow_color: Color::BLACK,
        highlight_color: Color::WHITE,
        blur_x: swf::Fixed16::from_f32(4.5),
        blur_y: swf::Fixed16::from_f32(3.5),
        angle: swf::Fixed16::from_f32(0.6),
        distance: swf::Fixed16::from_f32(3.25),
        strength: swf::Fixed8::ONE,
        flags: swf::BevelFilterFlags::from_passes(2) | swf::BevelFilterFlags::COMPOSITE_SOURCE,
    });
    let mut matrix = swf::ColorMatrixFilter::default();
    matrix.matrix[0] = 0.75;
    matrix.matrix[6] = 1.25;
    let filters = [
        vec![Filter::ColorMatrixFilter(matrix.clone())],
        vec![blur.clone()],
        vec![glow],
        vec![shadow],
        vec![bevel],
        vec![Filter::ColorMatrixFilter(matrix), blur],
    ];
    for quality in [StageQuality::Low, StageQuality::High] {
        for (filter_index, filters) in filters.iter().enumerate() {
            let mut renderer = WgpuRenderBackend::with_options(
                descriptors.clone(),
                TextureTarget::new(&descriptors.device, (96, 96)).unwrap(),
                false,
            )
            .unwrap();
            renderer.set_quality(quality).unwrap();
            let cache = renderer
                .create_bitmap_cache(96.try_into().unwrap(), 96.try_into().unwrap())
                .unwrap();
            for (step, x) in [1.25, 14.75, 27.35].into_iter().enumerate() {
                let scene = || {
                    let mut commands = CommandList::new();
                    rect(
                        &mut commands,
                        x,
                        16.25,
                        18.5,
                        13.75,
                        Color {
                            r: 171,
                            g: 80,
                            b: 41,
                            a: 173,
                        },
                    );
                    rect(
                        &mut commands,
                        24.25,
                        20.75,
                        9.5,
                        20.25,
                        Color {
                            r: 40,
                            g: 161,
                            b: 203,
                            a: 137,
                        },
                    );
                    commands
                };
                let render =
                    |renderer: &mut WgpuRenderBackend<TextureTarget>,
                     handle: ruffle_render::bitmap::BitmapHandle| {
                        let mut stage = CommandList::new();
                        stage.render_bitmap(
                            handle.clone(),
                            Default::default(),
                            false,
                            PixelSnapping::Never,
                            PixelRegion::for_whole_size(96, 96),
                        );
                        renderer
                            .submit_frame(
                                Color::TRANSPARENT,
                                stage,
                                vec![BitmapCacheEntry {
                                    handle,
                                    commands: scene(),
                                    clear: Color::TRANSPARENT,
                                    filters: filters.clone(),
                                }],
                            )
                            .unwrap();
                        renderer.capture_frame().unwrap().unwrap().into_raw()
                    };
                let actual = render(&mut renderer, cache.clone());
                let mut reference = WgpuRenderBackend::with_options(
                    descriptors.clone(),
                    TextureTarget::new(&descriptors.device, (96, 96)).unwrap(),
                    false,
                )
                .unwrap();
                reference.set_quality(quality).unwrap();
                let fresh = reference
                    .create_bitmap_cache(96.try_into().unwrap(), 96.try_into().unwrap())
                    .unwrap();
                let expected = render(&mut reference, fresh);
                let difference = actual
                    .iter()
                    .zip(&expected)
                    .enumerate()
                    .find(|(_, (a, b))| a != b);
                assert!(
                    difference.is_none(),
                    "filter {filter_index}, quality {quality:?}, step {step}: first differing byte {difference:?}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn native_preparation_parallel_shapes_match_serial_pixels_and_join_before_encoding() {
    use ruffle_render::{
        shape_utils::{DistilledShape, DrawCommand, DrawPath, FillRule},
        transform::Transform,
    };
    let descriptors = gpu();
    let mut images = Vec::new();
    for parallel in [false, true] {
        let target = TextureTarget::new(&descriptors.device, (64, 64)).unwrap();
        let mut renderer =
            WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
        let before = descriptors.runtime.preparation().stats().submitted;
        let mut commands = CommandList::new();
        for n in 0..20 {
            let style = swf::FillStyle::Color(Color::from_rgb(0x70a0ee + n * 33, 153));
            let point = |x, y| swf::Point::new(Twips::from_pixels(x), Twips::from_pixels(y));
            let handle = renderer
                .register_shape(
                    DistilledShape {
                        id: n as u16,
                        shape_bounds: Default::default(),
                        edge_bounds: Default::default(),
                        paths: vec![DrawPath::Fill {
                            style: &style,
                            winding_rule: FillRule::EvenOdd,
                            commands: vec![
                                DrawCommand::MoveTo(point(0.0, 0.0)),
                                DrawCommand::QuadraticCurveTo {
                                    control: point(8.5, -4.25),
                                    anchor: point(18.75, 1.0),
                                },
                                DrawCommand::LineTo(point(14.25, 19.5)),
                                DrawCommand::LineTo(point(0.0, 0.0)),
                            ],
                        }],
                    },
                    &crate::shape_recipe::FrozenBitmapSource::default(),
                )
                .unwrap();
            let mut transform = Transform::default();
            transform.matrix.tx = Twips::from_pixels(f64::from(n) * 1.25 + 0.25);
            transform.matrix.ty = Twips::from_pixels(f64::from(n) * 1.5 + 2.5);
            commands.render_shape(handle, transform);
        }
        if parallel {
            // Invisible, culled leaves cross the automatic work threshold without
            // changing pixels or introducing a production test switch.
            for _ in 0..256 {
                rect(&mut commands, 1000.0, 1000.0, 1.0, 1.0, Color::TRANSPARENT);
            }
        }
        renderer
            .submit_frame(Color::BLACK, commands, vec![])
            .unwrap();
        let stats = descriptors.runtime.preparation().stats();
        assert_eq!(stats.active, 0, "all CPU tasks joined in this frame");
        assert_eq!(stats.bytes, 0, "scratch permits returned after publication");
        assert_eq!(stats.submitted > before, parallel);
        assert!(
            stats.submitted - before <= 3,
            "small shapes share deterministic preparation partitions"
        );
        images.push(renderer.capture_frame().unwrap().unwrap().into_raw());
    }
    assert_eq!(images[0], images[1]);
    assert!(images[0].chunks_exact(4).any(|p| p != [0, 0, 0, 255]));
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn native_preparation_large_shape_uses_idle_worker_capacity_without_changing_pixels() {
    let bytes = include_bytes!(
        "../../../../tests/tests/swfs/visual/simple_shapes/heavy_tesselation/test.swf"
    );
    let stream = swf::decompress_swf(&bytes[..]).unwrap();
    let movie = swf::parse_swf(&stream).unwrap();
    let mut images = Vec::new();
    for staging in [64, 32, 16] {
        let descriptors = gpu_with_budgets(None, Some(staging * 1024 * 1024));
        let target = TextureTarget::new(&descriptors.device, (128, 128)).unwrap();
        let mut renderer =
            WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
        renderer.set_quality(StageQuality::Low).unwrap();
        let mut commands = CommandList::new();
        for tag in &movie.tags {
            if let swf::Tag::DefineShape(shape) = tag {
                let distilled = ruffle_render::shape_utils::DistilledShape::from(shape.as_ref());
                let handle = renderer
                    .register_shape(
                        distilled,
                        &crate::shape_recipe::FrozenBitmapSource::default(),
                    )
                    .unwrap();
                commands.render_shape(handle, Default::default());
            }
        }
        renderer
            .submit_frame(Color::WHITE, commands, vec![])
            .unwrap();
        let stats = descriptors.runtime.preparation().stats();
        assert_eq!((stats.active, stats.bytes), (0, 0));
        assert!(stats.peak_bytes <= staging * 1024 * 1024);
        images.push(renderer.capture_frame().unwrap().unwrap().into_raw());
    }
    assert_eq!(images[0], images[1]);
    assert_eq!(images[0], images[2]);
    assert!(images[0].chunks_exact(4).any(|p| p != [255, 255, 255, 255]));
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn native_preparation_resize_and_unload_cancel_captured_generations() {
    let descriptors = gpu();
    let target = TextureTarget::new_without_readback(&descriptors.device, (16, 16)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors.clone(), target, false).unwrap();
    let before_resize = renderer.preparation_generation.capture();
    renderer
        .set_viewport_dimensions(ruffle_render::backend::ViewportDimensions {
            width: 32,
            height: 32,
            scale_factor: 1.0,
        })
        .unwrap();
    let before_unload = renderer.preparation_generation.capture();
    drop(renderer);
    for generation in [before_resize, before_unload] {
        assert!(
            !descriptors
                .runtime
                .preparation()
                .for_each_batch(
                    &generation,
                    true,
                    [crate::prepare::Work {
                        value: 42,
                        bytes: 64
                    }],
                    |_| panic!("stale task must not run"),
                    |_: Vec<()>| panic!("stale output must not publish"),
                )
                .unwrap()
        );
    }
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn one_hundred_renderer_unloads_release_all_transient_resource_identities() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat, PixelRegion, PixelSnapping};
    use ruffle_render::transform::Transform;
    let descriptors = gpu();
    let cycle = || {
        let target = TextureTarget::new_without_readback(&descriptors.device, (64, 64)).unwrap();
        let mut renderer =
            WgpuRenderBackend::with_options(descriptors.clone(), target, true).unwrap();
        let bitmap = renderer
            .register_bitmap(Bitmap::new(
                16,
                16,
                BitmapFormat::Rgba,
                [90, 50, 20, 255].repeat(256),
            ))
            .unwrap();
        let mut children = CommandList::new();
        children.render_bitmap(
            bitmap,
            Transform::default(),
            true,
            PixelSnapping::Never,
            PixelRegion::for_whole_size(16, 16),
        );
        let mut commands = CommandList::new();
        commands.blend(children, RenderBlendMode::Builtin(BlendMode::Multiply));
        renderer
            .submit_frame(Color::WHITE, commands, vec![])
            .unwrap();
        descriptors
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(2)),
            })
            .unwrap();
        renderer.collect_gpu_samples();
        drop(renderer);
        descriptors.runtime.purge_idle_and_cached_resources();
        descriptors.runtime.check_health().unwrap();
    };
    cycle(); // Warm immutable descriptor/pipeline storage before measuring owners.
    let baseline: std::collections::HashSet<_> = descriptors
        .runtime
        .resources()
        .into_iter()
        .map(|r| r.id)
        .collect();
    let bytes = descriptors.runtime.stats().total_bytes;
    for cycle_index in 0..100 {
        cycle();
        let remaining = descriptors.runtime.resources();
        assert!(
            remaining.iter().all(|r| baseline.contains(&r.id)),
            "unloaded resources survived cycle {cycle_index}: {remaining:?}"
        );
        assert_eq!(descriptors.runtime.stats().total_bytes, bytes);
        assert_eq!(descriptors.runtime.stats().retiring_bytes, 0);
        assert_eq!(descriptors.runtime.stats().reserved_bytes, 0);
    }
}

#[test]
#[ignore = "requires a native GL adapter"]
fn native_gl_renders_geometry_and_uploaded_bitmaps_with_exact_pixels() {
    use ruffle_render::bitmap::{Bitmap, BitmapFormat, PixelRegion, PixelSnapping};
    let descriptors = gpu_with_backend(wgpu::Backends::GL, None, None);
    let target = TextureTarget::new(&descriptors.device, (4, 2)).unwrap();
    let mut renderer = WgpuRenderBackend::with_options(descriptors, target, false).unwrap();
    renderer.set_quality(StageQuality::Low).unwrap();
    let bitmap = renderer
        .register_bitmap(Bitmap::new(
            2,
            2,
            BitmapFormat::Rgba,
            [0, 0, 255, 255].repeat(4),
        ))
        .unwrap();
    for _ in 0..4 {
        let mut commands = CommandList::new();
        rect(&mut commands, 0.0, 0.0, 2.0, 2.0, Color::RED);
        let mut transform = ruffle_render::transform::Transform::default();
        transform.matrix.tx = Twips::from_pixels(2.0);
        commands.render_bitmap(
            bitmap.clone(),
            transform,
            false,
            PixelSnapping::Never,
            PixelRegion::for_whole_size(2, 2),
        );
        renderer
            .submit_frame(Color::WHITE, commands, vec![])
            .unwrap();
        let image = renderer.capture_frame().unwrap().unwrap();
        assert_eq!(
            image.get_pixel(0, 0).0,
            [255, 0, 0, 255],
            "geometry must reach the final attachment"
        );
        assert_eq!(
            image.get_pixel(2, 0).0,
            [0, 0, 255, 255],
            "uploaded bitmap must reach the final attachment"
        );
    }
}

#[test]
#[ignore = "requires a native GL adapter"]
fn native_gl_initialized_and_staged_buffers_preserve_bytes() {
    let descriptors = gpu_with_backend(wgpu::Backends::GL, None, None);
    let device = &descriptors.device;
    let initialized = device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[42; 16],
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::UNIFORM,
        })
        .unwrap();
    let output = device
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default()).unwrap();
    encoder.copy_buffer_to_buffer(&initialized, 0, &output, 0, 16);
    let mut arena = crate::frame_upload::FrameUploadArena::new(device.clone(), 64);
    arena
        .write_buffer(
            &mut encoder,
            &output,
            16,
            wgpu::BufferSize::new(16).unwrap(),
        )
        .unwrap()
        .copy_from_slice(&[73; 16]);
    arena.finish();
    let index = descriptors
        .queue
        .submit([encoder.finish().unwrap()])
        .unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    output.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(std::time::Duration::from_secs(2)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(2))
        .unwrap()
        .unwrap();
    let bytes = output.get_mapped_range(..).unwrap();
    assert_eq!(&bytes[..16], &[42; 16]);
    assert_eq!(&bytes[16..], &[73; 16]);
    drop(bytes);
    output.unmap();
}

#[test]
#[ignore = "requires a Vulkan adapter"]
fn pixelbender_allocations_are_attributed_to_the_shader_until_drop() {
    use crate::gpu_runtime::OwnerKind;
    let descriptors = gpu();
    let baseline: std::collections::HashSet<_> = descriptors
        .runtime
        .resources()
        .iter()
        .map(|r| r.id)
        .collect();
    let parsed = ruffle_render::pixel_bender::parse_shader(
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/tests/swfs/avm2/pixelbender_logicalnot/logicalnot.pbj"
        )),
        true,
    )
    .unwrap();
    let shader = crate::pixel_bender::PixelBenderWgpuShader::new(&descriptors, parsed).unwrap();
    let shader_resources: Vec<_> = descriptors
        .runtime
        .resources()
        .into_iter()
        .filter(|r| !baseline.contains(&r.id))
        .collect();
    assert!(!shader_resources.is_empty());
    assert!(
        shader_resources
            .iter()
            .all(|r| r.owner.kind == OwnerKind::ShaderJob),
        "shader allocations must be visible to the unload oracle: {shader_resources:?}"
    );
    assert!(
        shader_resources
            .iter()
            .all(|r| r.owner == shader_resources[0].owner)
    );
    drop(shader);
    assert!(
        descriptors
            .runtime
            .resources()
            .iter()
            .all(|r| baseline.contains(&r.id))
    );
}
