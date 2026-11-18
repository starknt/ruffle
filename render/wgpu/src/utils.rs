use crate::buffer_pool::BufferDescription;
use crate::descriptors::Descriptors;
use crate::gpu_resources::CommandEncoder;
use std::borrow::Cow;
use wgpu::TextureFormat;

macro_rules! create_debug_label {
    ($($arg:tt)*) => (
        if cfg!(feature = "render_debug_labels") {
            Some(format!($($arg)*))
        } else {
            None
        }
    )
}

pub fn remove_srgb(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    match format {
        wgpu::TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Bc1RgbaUnormSrgb => wgpu::TextureFormat::Bc1RgbaUnorm,
        wgpu::TextureFormat::Bc2RgbaUnormSrgb => wgpu::TextureFormat::Bc2RgbaUnorm,
        wgpu::TextureFormat::Bc3RgbaUnormSrgb => wgpu::TextureFormat::Bc3RgbaUnorm,
        wgpu::TextureFormat::Bc7RgbaUnormSrgb => wgpu::TextureFormat::Bc7RgbaUnorm,
        wgpu::TextureFormat::Etc2Rgb8UnormSrgb => wgpu::TextureFormat::Etc2Rgb8Unorm,
        wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb => wgpu::TextureFormat::Etc2Rgb8A1Unorm,
        wgpu::TextureFormat::Etc2Rgba8UnormSrgb => wgpu::TextureFormat::Etc2Rgba8Unorm,
        wgpu::TextureFormat::Astc {
            block,
            channel: wgpu::AstcChannel::UnormSrgb,
        } => wgpu::TextureFormat::Astc {
            block,
            channel: wgpu::AstcChannel::Unorm,
        },
        _ => format,
    }
}

pub fn format_list<'a>(values: &[&'a str], connector: &'a str) -> Cow<'a, str> {
    match values.len() {
        0 => Cow::Borrowed(""),
        1 => Cow::Borrowed(values[0]),
        _ => Cow::Owned(
            values[0..values.len() - 1].join(", ")
                + " "
                + connector
                + " "
                + values[values.len() - 1],
        ),
    }
}

pub fn get_backend_names(backends: wgpu::Backends) -> Vec<&'static str> {
    let mut names = Vec::new();

    if backends.contains(wgpu::Backends::VULKAN) {
        names.push("Vulkan");
    }
    if backends.contains(wgpu::Backends::DX12) {
        names.push("DirectX 12");
    }
    if backends.contains(wgpu::Backends::METAL) {
        names.push("Metal");
    }
    if backends.contains(wgpu::Backends::GL) {
        names.push("Open GL");
    }

    names
}

pub fn create_buffer_with_data(
    device: &crate::gpu_resources::Device,
    data: &[u8],
    usage: wgpu::BufferUsages,
    label: Option<String>,
) -> Result<crate::gpu_resources::Buffer, crate::gpu_runtime::RenderFailure> {
    Ok(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            usage,
            label: label.as_deref(),
            contents: data,
        })?,
    )
}

// Based off wgpu example 'capture'
#[derive(Debug, Clone)]
pub struct BufferDimensions {
    pub format: TextureFormat,
    pub width: usize,
    pub height: usize,
    pub unpadded_bytes_per_row: usize,
    pub padded_bytes_per_row: u32,
}

impl BufferDimensions {
    pub fn new(width: usize, height: usize, format: TextureFormat) -> Self {
        let bytes_per_pixel = format.block_copy_size(None).unwrap() as usize;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = (unpadded_bytes_per_row + padded_bytes_per_row_padding) as u32;

        Self {
            format,
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }

    pub fn size(&self) -> u64 {
        self.padded_bytes_per_row as u64 * self.height as u64
    }
}

impl BufferDescription for BufferDimensions {
    type Cost = u64;

    fn cost_to_use(&self, other: &Self) -> Option<Self::Cost> {
        if self.size() <= other.size() {
            Some(other.size() - self.size())
        } else {
            None
        }
    }
}

pub fn capture_image<R, F: FnOnce(&[u8], u32) -> R>(
    device: &crate::gpu_resources::Device,
    buffer: &crate::gpu_resources::Buffer,
    dimensions: &BufferDimensions,
    index: Option<wgpu::SubmissionIndex>,
    with_rgba: F,
) -> Result<R, crate::gpu_runtime::RenderFailure> {
    use crate::gpu_runtime::{RenderFailure, RenderFailureKind};
    let started = std::time::Instant::now();
    let deadline = std::time::Duration::from_secs(2);
    let (sender, receiver) = std::sync::mpsc::channel();
    let buffer_slice = buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let result = (|| {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: index,
                timeout: Some(deadline),
            })
            .map_err(poll_failure)?;
        receiver
            .recv_timeout(deadline.saturating_sub(started.elapsed()))
            .map_err(|error| {
                RenderFailure::new(
                    RenderFailureKind::Timeout,
                    format!("readback callback: {error}"),
                )
            })?
            .map_err(|error| {
                RenderFailure::new(
                    RenderFailureKind::MapFailed,
                    format!("readback map: {error}"),
                )
            })?;
        let map = buffer_slice.get_mapped_range().map_err(|error| {
            RenderFailure::new(
                RenderFailureKind::MapFailed,
                format!("readback range: {error}"),
            )
        })?;
        let result = with_rgba(&map, dimensions.padded_bytes_per_row);
        drop(map);
        Ok(result)
    })();
    buffer.unmap();
    result
}

pub fn poll_failure(error: wgpu::PollError) -> crate::gpu_runtime::RenderFailure {
    use crate::gpu_runtime::{RenderFailure, RenderFailureKind};
    let kind = match error {
        wgpu::PollError::Timeout => RenderFailureKind::Timeout,
        wgpu::PollError::WrongSubmissionIndex(..) => RenderFailureKind::Internal,
    };
    RenderFailure::new(kind, error.to_string())
}

pub fn buffer_to_image(
    device: &crate::gpu_resources::Device,
    buffer: &crate::gpu_resources::Buffer,
    dimensions: &BufferDimensions,
    index: Option<wgpu::SubmissionIndex>,
    size: wgpu::Extent3d,
) -> Result<image::RgbaImage, crate::gpu_runtime::RenderFailure> {
    capture_image(device, buffer, dimensions, index, |rgba, _buffer_width| {
        let mut bytes = Vec::with_capacity(dimensions.height * dimensions.unpadded_bytes_per_row);

        for chunk in rgba.chunks(dimensions.padded_bytes_per_row as usize) {
            bytes.extend_from_slice(&chunk[..dimensions.unpadded_bytes_per_row]);
        }

        // The image copied from the GPU uses premultiplied alpha, so
        // convert to straight alpha if requested by the user.
        ruffle_render::utils::unmultiply_alpha_rgba(&mut bytes);

        image::RgbaImage::from_raw(size.width, size.height, bytes)
            .expect("Retrieved texture buffer must be a valid RgbaImage")
    })
}

pub fn stage_sample_count_supported(
    adapter: &wgpu::Adapter,
    format: wgpu::TextureFormat,
    count: u32,
) -> bool {
    let color = adapter.get_texture_format_features(format).flags;
    let stencil = adapter
        .get_texture_format_features(wgpu::TextureFormat::Stencil8)
        .flags;
    color.sample_count_supported(count)
        && stencil.sample_count_supported(count)
        && (count == 1 || color.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE))
}

pub fn supported_sample_count(
    adapter: &wgpu::Adapter,
    mut sample_count: u32,
    format: wgpu::TextureFormat,
) -> u32 {
    let features = adapter.get_texture_format_features(format).flags;

    // Keep halving the sample count until we get one that's supported - or 1 (no multisampling)
    // It's not guaranteed that supporting 4x means supporting 2x, so there's no "max" option
    // And it's probably safer to round down than up, given it's a performance setting.
    while sample_count > 1 && !features.sample_count_supported(sample_count) {
        sample_count /= 2;
    }
    sample_count
}

pub fn run_copy_pipeline(
    descriptors: &Descriptors,
    format: wgpu::TextureFormat,
    frame_view: &crate::gpu_resources::TextureView,
    input: &crate::gpu_resources::TextureView,
    sample_count: u32,
    encoder: &mut CommandEncoder,
) -> Result<(), crate::gpu_runtime::RenderFailure> {
    run_copy_regions_pipeline(
        descriptors,
        format,
        frame_view,
        input,
        sample_count,
        encoder,
        None,
    )
}

pub fn run_copy_regions_pipeline(
    descriptors: &Descriptors,
    format: wgpu::TextureFormat,
    frame_view: &crate::gpu_resources::TextureView,
    input: &crate::gpu_resources::TextureView,
    sample_count: u32,
    encoder: &mut CommandEncoder,
    regions: Option<&[ruffle_render::region::RenderRect]>,
) -> Result<(), crate::gpu_runtime::RenderFailure> {
    let copy_bind_group = descriptors.cached_bind_group(&wgpu::BindGroupDescriptor {
        layout: &descriptors.bind_layouts.bitmap,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(input),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(
                    descriptors.bitmap_samplers.get_sampler(false, false),
                ),
            },
        ],
        label: create_debug_label!("Copy bind group").as_deref(),
    })?;

    let pipeline = descriptors.copy_pipeline(format, sample_count)?;

    // We overwrite the pixels in the target texture (no blending at all),
    // so this doesn't matter.
    let load = if regions.is_some() {
        wgpu::LoadOp::Load
    } else {
        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
    };

    ruffle_render::profile::work(|w| {
        w.render_passes += 1;
        w.draw_calls += regions.map_or(1, |regions| regions.len() as u64);
        w.pipeline_changes += 1;
        w.material_binds += 1;
        w.pass_pixels +=
            u64::from(frame_view.texture().width()) * u64::from(frame_view.texture().height());
    });
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: create_debug_label!("Copy back to render target").as_deref(),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: frame_view,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
            resolve_target: None,
            depth_slice: None,
        })],
        ..Default::default()
    });

    render_pass.set_pipeline(&pipeline);
    render_pass.set_bind_group(2, (&copy_bind_group).raw(), &[]);

    if let Some(regions) = regions {
        for region in regions {
            render_pass.set_scissor_rect(region.x, region.y, region.width, region.height);
            render_pass.draw(0..3, 0..1);
        }
    } else {
        render_pass.draw(0..3, 0..1);
    }
    drop(render_pass);

    Ok(())
}

#[derive(Debug)]
pub struct SampleCountMap<T> {
    one: T,
    two: T,
    four: T,
    eight: T,
    sixteen: T,
}

impl<T: Default> Default for SampleCountMap<T> {
    fn default() -> Self {
        SampleCountMap {
            one: Default::default(),
            two: Default::default(),
            four: Default::default(),
            eight: Default::default(),
            sixteen: Default::default(),
        }
    }
}

impl<T> SampleCountMap<T> {
    pub fn get(&self, sample_count: u32) -> &T {
        match sample_count {
            1 => &self.one,
            2 => &self.two,
            4 => &self.four,
            8 => &self.eight,
            16 => &self.sixteen,
            _ => unreachable!("Sample counts must be powers of two between 1..=16"),
        }
    }
}

impl<T> SampleCountMap<std::sync::OnceLock<T>> {
    pub fn get_or_init<F, E>(&self, sample_count: u32, init: F) -> Result<&T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        let cell = self.get(sample_count);
        if let Some(value) = cell.get() {
            return Ok(value);
        }
        let value = init()?;
        Ok(cell.get_or_init(|| value))
    }
}
