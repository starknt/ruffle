use crate::blend::ComplexBlend;
use crate::surface::buffers::transforms_per_draw;
use enum_map::{EnumMap, enum_map};
use ruffle_render::shader_source::SHADER_FILTER_COMMON;

#[derive(Debug)]
pub struct Shaders {
    pub color_shader: crate::gpu_resources::ShaderModule,
    /// This has a pipeline-overridable `bool` constant, `late_saturate`,
    /// with a default of `false`. It switches to performing saturation
    /// after re-multiplying the alpha, rather than before. This is used
    /// for the Stage3D `bitmap_opaque` pipeline, which needs to be able to
    /// avoid changing initially-in-range rgb values (regadless of whether
    /// dividing by the alpha value would produce an out-of-range value).
    pub bitmap_shader: crate::gpu_resources::ShaderModule,
    pub bitmap_batch_shader: crate::gpu_resources::ShaderModule,
    pub gradient_shader: crate::gpu_resources::ShaderModule,
    pub copy_shader: crate::gpu_resources::ShaderModule,
    pub alpha_mask_shader: crate::gpu_resources::ShaderModule,
    pub blend_shaders: EnumMap<ComplexBlend, crate::gpu_resources::ShaderModule>,
    pub color_matrix_filter: crate::gpu_resources::ShaderModule,
    pub convolution_filter: crate::gpu_resources::ShaderModule,
    pub blur_filter: crate::gpu_resources::ShaderModule,
    pub glow_filter: crate::gpu_resources::ShaderModule,
    pub gradient_glow_filter: crate::gpu_resources::ShaderModule,
    pub bevel_filter: crate::gpu_resources::ShaderModule,
    pub displacement_map_filter: crate::gpu_resources::ShaderModule,
}

impl Shaders {
    pub(crate) fn new(
        device: &crate::gpu_resources::Device,
        bindings: crate::capabilities::BitmapBindings,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let max_transforms = transforms_per_draw(&device.limits());
        let color_shader = make_shader(
            device,
            "color.wgsl",
            include_str!("../shaders/color.wgsl"),
            max_transforms,
        )?;
        let bitmap_shader = make_shader(
            device,
            "bitmap.wgsl",
            include_str!("../shaders/bitmap.wgsl"),
            max_transforms,
        )?;
        use crate::capabilities::BitmapBindings;
        let bitmap_sampling = match bindings {
            BitmapBindings::Array => include_str!("../shaders/bitmap_batch_binding_array.wgsl"),
            BitmapBindings::Layers => include_str!("../shaders/bitmap_batch_layers.wgsl"),
            _ => include_str!("../shaders/bitmap_batch_explicit.wgsl"),
        };
        let bitmap_batch_shader = make_shader_with_extensions(
            device,
            "bitmap_batch.wgsl",
            &format!(
                "{bitmap_sampling}\n{}",
                include_str!("../shaders/bitmap_batch.wgsl")
            ),
            max_transforms,
            if bindings == BitmapBindings::Array {
                "enable wgpu_binding_array;"
            } else {
                ""
            },
        )?;
        let copy_shader = make_shader(
            device,
            "copy.wgsl",
            include_str!("../shaders/copy.wgsl"),
            max_transforms,
        )?;
        let color_matrix_filter = make_filter_shader(
            device,
            "filter/color_matrix.wgsl",
            include_str!("../shaders/filter/color_matrix.wgsl"),
        )?;
        let convolution_filter = make_filter_shader(
            device,
            "filter/convolution.wgsl",
            include_str!("../shaders/filter/convolution.wgsl"),
        )?;
        let blur_filter = make_filter_shader(
            device,
            "filter/blur.wgsl",
            include_str!("../shaders/filter/blur.wgsl"),
        )?;
        let glow_filter = make_filter_shader(
            device,
            "filter/glow.wgsl",
            include_str!("../shaders/filter/glow.wgsl"),
        )?;
        let gradient_glow_filter = make_filter_shader(
            device,
            "filter/gradient_glow.wgsl",
            include_str!("../shaders/filter/gradient_glow.wgsl"),
        )?;
        let bevel_filter = make_filter_shader(
            device,
            "filter/bevel.wgsl",
            include_str!("../shaders/filter/bevel.wgsl"),
        )?;
        let displacement_map_filter = make_filter_shader(
            device,
            "filter/displacement_map.wgsl",
            include_str!("../shaders/filter/displacement_map.wgsl"),
        )?;
        let gradient_shader = make_shader(
            device,
            "gradient.wgsl",
            include_str!("../shaders/gradient.wgsl"),
            max_transforms,
        )?;
        let alpha_mask_shader = make_shader(
            device,
            "alpha_mask.wgsl",
            include_str!("../shaders/alpha_mask.wgsl"),
            max_transforms,
        )?;

        let blend_shaders = enum_map! {
            ComplexBlend::Multiply => make_shader(device, "blend/multiply.wgsl", include_str!("../shaders/blend/multiply.wgsl"), max_transforms)?,
            ComplexBlend::Lighten => make_shader(device, "blend/lighten.wgsl", include_str!("../shaders/blend/lighten.wgsl"), max_transforms)?,
            ComplexBlend::Darken => make_shader(device, "blend/darken.wgsl", include_str!("../shaders/blend/darken.wgsl"), max_transforms)?,
            ComplexBlend::Difference => make_shader(device, "blend/difference.wgsl", include_str!("../shaders/blend/difference.wgsl"), max_transforms)?,
            ComplexBlend::Invert => make_shader(device, "blend/invert.wgsl", include_str!("../shaders/blend/invert.wgsl"), max_transforms)?,
            ComplexBlend::Alpha => make_shader(device, "blend/alpha.wgsl", include_str!("../shaders/blend/alpha.wgsl"), max_transforms)?,
            ComplexBlend::Erase => make_shader(device, "blend/erase.wgsl", include_str!("../shaders/blend/erase.wgsl"), max_transforms)?,
            ComplexBlend::Overlay => make_shader(device, "blend/overlay.wgsl", include_str!("../shaders/blend/overlay.wgsl"), max_transforms)?,
            ComplexBlend::HardLight => make_shader(device, "blend/hardlight.wgsl", include_str!("../shaders/blend/hardlight.wgsl"), max_transforms)?,
        };

        Ok(Self {
            color_shader,
            bitmap_shader,
            bitmap_batch_shader,
            gradient_shader,
            copy_shader,
            alpha_mask_shader,
            blend_shaders,
            color_matrix_filter,
            convolution_filter,
            blur_filter,
            glow_filter,
            gradient_glow_filter,
            bevel_filter,
            displacement_map_filter,
        })
    }
}

fn make_shader(
    device: &crate::gpu_resources::Device,
    name: &str,
    source: &str,
    max_transforms: u64,
) -> Result<crate::gpu_resources::ShaderModule, crate::gpu_runtime::RenderFailure> {
    make_shader_with_extensions(device, name, source, max_transforms, "")
}

fn make_shader_with_extensions(
    device: &crate::gpu_resources::Device,
    name: &str,
    source: &str,
    max_transforms: u64,
    extensions: &str,
) -> Result<crate::gpu_resources::ShaderModule, crate::gpu_runtime::RenderFailure> {
    let common = include_str!("../shaders/common.wgsl");
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: create_debug_label!("Shader {name}").as_deref(),
        source: wgpu::ShaderSource::Wgsl(
            format!("{extensions}\n{common}\nconst max_transforms = {max_transforms};\n{source}")
                .into(),
        ),
    })
}
fn make_filter_shader(
    device: &crate::gpu_resources::Device,
    name: &str,
    source: &str,
) -> Result<crate::gpu_resources::ShaderModule, crate::gpu_runtime::RenderFailure> {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: create_debug_label!("Shader {name}").as_deref(),
        source: wgpu::ShaderSource::Wgsl(format!("{SHADER_FILTER_COMMON}\n{source}").into()),
    })
}
