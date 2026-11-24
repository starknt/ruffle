//! Native device negotiation and the features actually available to a render path.
use wgpu::{
    Adapter, Device, DownlevelFlags, Features, Limits, TextureFormat, TextureFormatFeatures,
    TextureUsages,
};

#[derive(Clone, Copy)]
pub enum NativeGpuUse {
    Standalone,
    SharedStage,
}

pub(crate) fn compute_filters_requested() -> bool {
    std::env::var_os("RUFFLE_COMPUTE_FILTERS").as_deref() == Some("1".as_ref())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BitmapDraw {
    Direct,
    Control,
    Indirect,
    Cull,
    Count,
}

impl BitmapDraw {
    pub(crate) fn requested() -> Self {
        match std::env::var_os("RUFFLE_BITMAP_DRAW").as_deref() {
            Some(v) if v == "indirect" => Self::Indirect,
            Some(v) if v == "cull" => Self::Cull,
            Some(v) if v == "count" => Self::Count,
            Some(v) if v == "direct-control" => Self::Control,
            _ => Self::Direct,
        }
    }

    fn can_request(self, adapter: &Adapter) -> bool {
        self != Self::Direct
            && adapter.get_info().backend == wgpu::Backend::Vulkan
            && adapter
                .get_downlevel_capabilities()
                .flags
                .contains(DownlevelFlags::INDIRECT_EXECUTION | DownlevelFlags::COMPUTE_SHADERS)
            && adapter.limits().max_storage_buffers_per_shader_stage >= 2
    }

    pub(crate) fn granted(adapter: &Adapter, device: &Device) -> Self {
        let mode = Self::requested();
        let limits = device.limits();
        if !mode.can_request(adapter)
            || mode == Self::Control
            || limits.max_storage_buffers_per_shader_stage < 2
            || limits.max_storage_buffer_binding_size < 65536
            || limits.max_compute_invocations_per_workgroup == 0
            || limits.max_compute_workgroup_size_x == 0
            || limits.max_compute_workgroup_size_y == 0
            || limits.max_compute_workgroup_size_z == 0
            || limits.max_compute_workgroups_per_dimension == 0
        {
            return Self::Direct;
        }
        if mode.culls()
            && !device
                .features()
                .contains(Features::INDIRECT_FIRST_INSTANCE)
            || mode == Self::Count
                && !device
                    .features()
                    .contains(Features::MULTI_DRAW_INDIRECT_COUNT)
        {
            return Self::Direct;
        }
        mode
    }

    pub(crate) fn culls(self) -> bool {
        matches!(self, Self::Cull | Self::Count)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BitmapBindings {
    Explicit,
    ExplicitOnArrayDevice,
    Array,
    Layers,
}

impl BitmapBindings {
    pub(crate) fn requested() -> Self {
        Self::select(std::env::var_os("RUFFLE_BITMAP_BINDINGS").as_deref())
    }

    fn select(setting: Option<&std::ffi::OsStr>) -> Self {
        match setting {
            Some(value) if value == "array" => Self::Array,
            Some(value) if value == "layers" => Self::Layers,
            Some(value) if value == "explicit-on-array-device" => Self::ExplicitOnArrayDevice,
            _ => Self::Explicit,
        }
    }

    const ARRAY_FEATURES: Features = Features::TEXTURE_BINDING_ARRAY
        .union(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

    pub(crate) fn supported(features: Features, limits: &Limits) -> bool {
        let count = crate::layouts::BITMAP_BATCH_TEXTURES as u32;
        features.contains(Self::ARRAY_FEATURES)
            && limits.max_binding_array_elements_per_shader_stage >= count
            && limits.max_sampled_textures_per_shader_stage >= count
    }
}

/// Preserve each host's supported limit profile while selecting optional features in one place.
pub fn device_descriptor(
    adapter: &Adapter,
    usage: NativeGpuUse,
) -> wgpu::DeviceDescriptor<'static> {
    let available = adapter.limits();
    let array_size = crate::layouts::BITMAP_BATCH_TEXTURES as u32;
    // A paired explicit control negotiates the same device as the candidate.
    let arrays = matches!(
        BitmapBindings::requested(),
        BitmapBindings::Array | BitmapBindings::Layers | BitmapBindings::ExplicitOnArrayDevice
    ) && BitmapBindings::supported(adapter.features(), &available);
    let (mut wanted, mut limits) = match usage {
        NativeGpuUse::Standalone => {
            let mut baseline = Limits::downlevel_webgl2_defaults()
                .using_resolution(available.clone())
                .using_alignment(available.clone());
            baseline.max_uniform_buffer_binding_size = available.max_uniform_buffer_binding_size;
            baseline.max_inter_stage_shader_variables = available.max_inter_stage_shader_variables;
            baseline.max_color_attachments = 4.min(available.max_color_attachments);
            if compute_filters_requested()
                && adapter
                    .get_downlevel_capabilities()
                    .flags
                    .contains(DownlevelFlags::COMPUTE_SHADERS)
            {
                baseline.max_storage_textures_per_shader_stage =
                    1.min(available.max_storage_textures_per_shader_stage);
                baseline.max_compute_invocations_per_workgroup =
                    64.min(available.max_compute_invocations_per_workgroup);
                baseline.max_compute_workgroup_size_x =
                    8.min(available.max_compute_workgroup_size_x);
                baseline.max_compute_workgroup_size_y =
                    8.min(available.max_compute_workgroup_size_y);
                baseline.max_compute_workgroup_size_z =
                    1.min(available.max_compute_workgroup_size_z);
                baseline.max_compute_workgroups_per_dimension =
                    available.max_compute_workgroups_per_dimension;
            }
            (
                Features::TEXTURE_COMPRESSION_BC
                    | crate::gpu_profiler::GpuProfiler::ALL_WGPU_TIMER_FEATURES,
                baseline,
            )
        }
        NativeGpuUse::SharedStage => (
            Features::TEXTURE_FORMAT_16BIT_NORM
                | Features::TIMESTAMP_QUERY
                | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
            available,
        ),
    };
    if BitmapDraw::requested().can_request(adapter) {
        let available = adapter.limits();
        // Match both sides of an experiment, including the optional count path.
        wanted |= Features::INDIRECT_FIRST_INSTANCE | Features::MULTI_DRAW_INDIRECT_COUNT;
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(2);
        limits.max_storage_buffer_binding_size = limits
            .max_storage_buffer_binding_size
            .max(65536.min(available.max_storage_buffer_binding_size));
        limits.max_compute_invocations_per_workgroup =
            limits.max_compute_invocations_per_workgroup.max(1);
        limits.max_compute_workgroup_size_x = limits.max_compute_workgroup_size_x.max(1);
        limits.max_compute_workgroup_size_y = limits.max_compute_workgroup_size_y.max(1);
        limits.max_compute_workgroup_size_z = limits.max_compute_workgroup_size_z.max(1);
        limits.max_compute_workgroups_per_dimension =
            available.max_compute_workgroups_per_dimension;
    }
    if arrays {
        wanted |= BitmapBindings::ARRAY_FEATURES;
        limits.max_binding_array_elements_per_shader_stage = limits
            .max_binding_array_elements_per_shader_stage
            .max(array_size);
    }
    wanted |= Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | Features::FLOAT32_FILTERABLE;
    wgpu::DeviceDescriptor {
        label: Some(match usage {
            NativeGpuUse::Standalone => "Ruffle renderer",
            NativeGpuUse::SharedStage => "Aola shared stage",
        }),
        required_features: wanted & adapter.features(),
        required_limits: limits,
        ..Default::default()
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeCapabilities {
    pub compute_rgba8: bool,
    pub filterable_rgba16f_storage: bool,
    pub max_compute_groups: u32,
    pub timestamps: bool,
    pub encoder_timestamps: bool,
    pub texture_binding_arrays: bool,
    pub non_uniform_texture_index: bool,
    pub indirect_draw: bool,
    pub indirect_first_instance: bool,
    pub indirect_count: bool,
    pub shader_f16: bool,
    pub subgroups: bool,
    pub pipeline_cache: bool,
}

impl NativeCapabilities {
    pub fn granted(adapter: &Adapter, device: &Device) -> Self {
        let features = device.features();
        let limits = device.limits();
        let downlevel = adapter.get_downlevel_capabilities().flags;
        let format_features = |format: TextureFormat| -> TextureFormatFeatures {
            if features.contains(Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
                adapter.get_texture_format_features(format)
            } else {
                format.guaranteed_format_features(features)
            }
        };
        let usage = TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
        let compute = downlevel.contains(DownlevelFlags::COMPUTE_SHADERS)
            && limits.max_storage_textures_per_shader_stage > 0
            && limits.max_compute_invocations_per_workgroup >= 64
            && limits.max_compute_workgroup_size_x >= 8
            && limits.max_compute_workgroup_size_y >= 8
            && limits.max_compute_workgroup_size_z >= 1
            && limits.max_compute_workgroups_per_dimension > 0;
        let fp16 = format_features(TextureFormat::Rgba16Float);
        let arrays = features.contains(Features::TEXTURE_BINDING_ARRAY)
            && limits.max_binding_array_elements_per_shader_stage > 0;
        let indirect = downlevel.contains(DownlevelFlags::INDIRECT_EXECUTION);
        Self {
            compute_rgba8: compute
                && format_features(TextureFormat::Rgba8Unorm)
                    .allowed_usages
                    .contains(usage),
            filterable_rgba16f_storage: compute
                && fp16.allowed_usages.contains(usage)
                && fp16
                    .flags
                    .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE),
            max_compute_groups: limits.max_compute_workgroups_per_dimension,
            timestamps: features.contains(Features::TIMESTAMP_QUERY),
            encoder_timestamps: features.contains(Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
            texture_binding_arrays: arrays,
            non_uniform_texture_index: arrays
                && features.contains(
                    Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                ),
            indirect_draw: indirect,
            indirect_first_instance: indirect
                && features.contains(Features::INDIRECT_FIRST_INSTANCE),
            indirect_count: indirect && features.contains(Features::MULTI_DRAW_INDIRECT_COUNT),
            shader_f16: features.contains(Features::SHADER_F16),
            subgroups: features.contains(Features::SUBGROUP),
            pipeline_cache: features.contains(Features::PIPELINE_CACHE),
        }
    }
}

#[cfg(test)]
mod bitmap_binding_tests {
    use super::*;

    #[test]
    fn native_backends_use_explicit_bindings_unless_arrays_are_requested() {
        assert_eq!(BitmapBindings::select(None), BitmapBindings::Explicit);
        assert_eq!(
            BitmapBindings::select(Some("array".as_ref())),
            BitmapBindings::Array
        );
        assert_eq!(
            BitmapBindings::select(Some("explicit".as_ref())),
            BitmapBindings::Explicit
        );
        assert_eq!(
            BitmapBindings::select(Some("explicit-on-array-device".as_ref())),
            BitmapBindings::ExplicitOnArrayDevice
        );
    }
}
