use anyhow::{Context, Result, ensure};
use image::RgbaImage;
use ruffle_render_wgpu::{backend::WgpuRenderBackend, target::TextureTarget};
use ruffle_test_framework::{
    environment::Environment,
    options::TestOptions,
    runner::TestStatus,
    test::Test,
    vfs::{PhysicalFS, VfsPath},
};
use std::any::Any;

pub fn flash_semantics(env: &impl Environment) -> Result<(), libtest_mimic::Failed> {
    let root = VfsPath::new(PhysicalFS::new("tests/stage_render_extent"));
    let test = Test::from_options(
        TestOptions::read(&root.join("test.toml")?)?,
        root,
        "stage_render_extent".into(),
    )?;
    let capture = |percent, width, height| -> Result<RgbaImage> {
        let mut runner = test.create_test_runner(env)?;
        {
            let mut player = runner.player().lock().unwrap();
            let renderer =
                <dyn Any>::downcast_mut::<WgpuRenderBackend<TextureTarget>>(player.renderer_mut())
                    .context("stage render extent regression requires a native GPU")?;
            renderer.set_stage_render_scale_percent(percent)?;
        }
        // tick() renders through Player and checks the movie's logical readbacks against output.txt.
        while runner.tick()? != TestStatus::Finished {}
        let mut player = runner.player().lock().unwrap();
        let renderer =
            <dyn Any>::downcast_mut::<WgpuRenderBackend<TextureTarget>>(player.renderer_mut())
                .unwrap();
        ensure!(
            renderer.last_stage_attachment() == Some((width, height, 4)),
            "{percent}% stage attachment {:?}",
            renderer.last_stage_attachment()
        );
        let image = renderer
            .capture_frame()?
            .context("missing resolved stage")?;
        ensure!(image.dimensions() == (width, height));
        Ok(image)
    };

    let full = capture(100, 105, 73)?;
    let blue = [0, 0, 255, 255];
    for (x, y, expected) in [
        (2, 3, [0, 0, 0, 255]),
        (80, 52, [255, 255, 0, 255]), // Scissored Stage3D draw at a nonzero stage origin.
        (50, 20, blue),               // Mask cuts the top of the filtered rectangle.
        (73, 30, blue),               // scrollRect cuts its right side.
        (48, 32, [255, 0, 255, 255]), // Difference blend of the red cache and blue background.
    ] {
        let actual = full.get_pixel(x, y).0;
        assert!(
            actual
                .into_iter()
                .zip(expected)
                .all(|(a, b)| a.abs_diff(b) <= 2),
            "100% stage ({x}, {y}): {actual:?}, expected {expected:?}"
        );
    }
    for (x, y) in [(35, 32), (51, 51)] {
        let [r, g, b, a] = full.get_pixel(x, y).0;
        assert!(
            (1..255).contains(&r) && (g, b, a) == (0, 255, 255),
            "missing blur fringe at ({x}, {y}): {r},{g},{b},{a}"
        );
    }
    for (percent, w, h) in [(75, 79, 55), (50, 53, 37)] {
        let scaled = capture(percent, w, h)?;
        compare_projection(&full, &scaled)?;
    }
    Ok(())
}

fn compare_projection(full: &RgbaImage, scaled: &RgbaImage) -> Result<()> {
    let (w, h) = scaled.dimensions();
    // A resolved raster sample can fall on either side of a hard edge. Require each
    // channel to lie within its projected source footprint, including both blur fringes.
    for (x, y, pixel) in scaled.enumerate_pixels() {
        let mut lo = [255u8; 4];
        let mut hi = [0u8; 4];
        for sy in y * full.height() / h..((y + 1) * full.height()).div_ceil(h) {
            for sx in x * full.width() / w..((x + 1) * full.width()).div_ceil(w) {
                for (c, value) in full.get_pixel(sx, sy).0.into_iter().enumerate() {
                    lo[c] = lo[c].min(value);
                    hi[c] = hi[c].max(value);
                }
            }
        }
        for (c, value) in pixel.0.into_iter().enumerate() {
            ensure!(
                value >= lo[c].saturating_sub(1) && value <= hi[c].saturating_add(1),
                "{w}x{h} ({x}, {y}) channel {c}: {value} outside projected {}..={}",
                lo[c],
                hi[c]
            );
        }
    }
    Ok(())
}
