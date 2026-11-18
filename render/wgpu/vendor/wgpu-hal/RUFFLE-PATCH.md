# Ruffle native GL patch

Vendored from the crates.io `wgpu-hal` 30.0.1 package (upstream commit
`40f4a34ebaf56f9a046231f54125ad046239d3f3`). Keep its MIT/Apache-2.0 licenses and version.

The sole engine change is in `src/gles/adapter.rs`: derive
`SHADER_BINDING_LAYOUT` from Naga's `Version::supports_explicit_locations()`.
A desktop GL 3.3 driver may advertise compute extensions while Naga emits GLSL
330 without explicit bindings. Using compute support to skip program remapping
leaves every uniform block/sampler at binding 0 and renders blank images.

Regression: `surface::tests::native_gl_renders_geometry_and_uploaded_bitmaps_with_exact_pixels`
in `render/wgpu/src/surface/tests.rs`, explicitly run with `--ignored` on native GL.
The original crate fails exact red/blue pixels; this patch must pass without
changing shader arithmetic, pixel tolerances or GPU backends. Remove this vendor
patch when an adopted upstream version includes the same capability correction.
