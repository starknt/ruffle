use crate::buffer_pool::PoolEntry;
use crate::gpu_runtime::{RenderFailure as Error, RenderFailureKind};
use crate::utils::{BufferDimensions, remove_srgb};
use ruffle_render::bitmap::PixelRegion;
use ruffle_render::error::RendererHealth;
use std::fmt::Debug;
use std::ops::Deref;
use tracing::instrument;

pub trait RenderTargetFrame: Debug {
    fn into_view(self) -> crate::gpu_resources::TextureView;

    fn view(&self) -> &crate::gpu_resources::TextureView;
}

pub trait RenderTarget: Debug + 'static {
    type Frame: RenderTargetFrame;

    fn resize(
        &mut self,
        device: &crate::gpu_resources::Device,
        width: u32,
        height: u32,
    ) -> Result<(), Error>;

    fn format(&self) -> wgpu::TextureFormat;

    fn width(&self) -> u32;

    fn height(&self) -> u32;

    /// Returns the frame to render into, or `None` if no frame could be
    /// acquired - in which case the caller should skip this frame, and
    /// may attempt to recreate the swap chain via [`RenderTarget::resize`].
    fn get_next_texture(&mut self) -> Result<Option<Self::Frame>, Error>;

    fn submit<I: IntoIterator<Item = crate::gpu_resources::CommandBuffer>>(
        &self,
        device: &crate::gpu_resources::Device,
        queue: &crate::gpu_resources::Queue,
        health: &RendererHealth,
        command_buffers: I,
        frame: Self::Frame,
    ) -> Result<wgpu::SubmissionIndex, Error>;
}

#[derive(Debug)]
pub struct SwapChainTarget {
    window_surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: crate::gpu_resources::Device,
}

#[derive(Debug)]
pub struct SwapChainTargetFrame {
    texture: wgpu::SurfaceTexture,
    view: crate::gpu_resources::TextureView,
}

impl RenderTargetFrame for SwapChainTargetFrame {
    fn into_view(self) -> crate::gpu_resources::TextureView {
        self.view
    }

    fn view(&self) -> &crate::gpu_resources::TextureView {
        &self.view
    }
}

impl SwapChainTarget {
    pub fn new(
        surface: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        (width, height): (u32, u32),
        device: &crate::gpu_resources::Device,
    ) -> Result<Self, Error> {
        if width == 0
            || height == 0
            || width > device.limits().max_texture_dimension_2d
            || height > device.limits().max_texture_dimension_2d
        {
            return Err(Error::new(
                RenderFailureKind::Validation,
                "invalid surface dimensions",
            ));
        }
        // Flash colors and blending are done in sRGB space -- we don't want the GPU to
        // adjust the colors. We prefer a non-sRGB surface format directly, but on platforms
        // that only expose sRGB surface formats we view the surface texture as its non-sRGB
        // counterpart so writes skip the linear->sRGB encode step.
        let capabilities = surface.get_capabilities(adapter);
        let format = capabilities
            .formats
            .iter()
            .find(|format| {
                matches!(
                    format,
                    wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
                )
            })
            .or_else(|| capabilities.formats.first())
            .copied()
            .ok_or_else(|| {
                Error::new(
                    RenderFailureKind::Validation,
                    "surface has no supported texture format",
                )
            })?;

        let linear_format = remove_srgb(format);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: capabilities.alpha_modes.first().copied().ok_or_else(|| {
                Error::new(
                    RenderFailureKind::Validation,
                    "surface has no supported alpha mode",
                )
            })?,
            view_formats: if linear_format == format {
                vec![format]
            } else {
                vec![format, linear_format]
            },
        };
        device.configure_surface(&surface, &surface_config)?;
        Ok(Self {
            surface_config,
            window_surface: surface,
            device: device.clone(),
        })
    }
}

impl RenderTarget for SwapChainTarget {
    type Frame = SwapChainTargetFrame;

    fn resize(
        &mut self,
        device: &crate::gpu_resources::Device,
        width: u32,
        height: u32,
    ) -> Result<(), Error> {
        self.surface_config.width = width;
        self.surface_config.height = height;
        device.configure_surface(&self.window_surface, &self.surface_config)?;

        Ok(())
    }

    fn format(&self) -> wgpu::TextureFormat {
        remove_srgb(self.surface_config.format)
    }

    fn width(&self) -> u32 {
        self.surface_config.width
    }

    fn height(&self) -> u32 {
        self.surface_config.height
    }

    fn get_next_texture(&mut self) -> Result<Option<Self::Frame>, Error> {
        let texture = match self.window_surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            // The texture is still usable - no reason to waste an already
            // acquired frame. Ideally the surface should also be reconfigured
            // before the next acquire; we don't currently do that (matching
            // pre-wgpu-29 behavior, which never checked the suboptimal flag).
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err(Error::new(RenderFailureKind::DeviceLost, "surface lost"));
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(Error::new(
                    RenderFailureKind::Validation,
                    "surface acquisition failed",
                ));
            }
            state => {
                tracing::warn!("Couldn't acquire surface texture: {:?}", state);
                return Ok(None);
            }
        };
        let view_format = remove_srgb(self.surface_config.format);

        let tracked = self.device.import_texture(
            texture.texture.clone(),
            crate::gpu_runtime::ResourceKind::HostTarget,
        )?;
        let view = tracked.create_view(&wgpu::TextureViewDescriptor {
            format: Some(view_format),
            ..Default::default()
        })?;

        Ok(Some(SwapChainTargetFrame { texture, view }))
    }

    #[instrument(level = "debug", skip_all)]
    fn submit<I: IntoIterator<Item = crate::gpu_resources::CommandBuffer>>(
        &self,
        _device: &crate::gpu_resources::Device,
        queue: &crate::gpu_resources::Queue,
        health: &RendererHealth,
        command_buffers: I,
        frame: Self::Frame,
    ) -> Result<wgpu::SubmissionIndex, Error> {
        let index = {
            let _timer = ruffle_render::profile::native_time(
                ruffle_render::profile::NativePhase::QueueSubmit,
            );
            health.check()?;
            queue.submit(command_buffers)?
        };
        let _timer =
            ruffle_render::profile::native_time(ruffle_render::profile::NativePhase::Present);
        health.check()?;
        queue.present(frame.texture);
        health.check()?;
        Ok(index)
    }
}

#[derive(Debug)]
pub enum MaybeOwnedBuffer {
    Borrowed(
        PoolEntry<crate::gpu_resources::Buffer, BufferDimensions>,
        BufferDimensions,
    ),
    Owned(crate::gpu_resources::Buffer, BufferDimensions),
}

impl MaybeOwnedBuffer {
    pub fn inner(&self) -> (&crate::gpu_resources::Buffer, &BufferDimensions) {
        match &self {
            MaybeOwnedBuffer::Borrowed(entry, dimensions) => ((*entry).deref(), dimensions),
            MaybeOwnedBuffer::Owned(buffer, dimensions) => (buffer, dimensions),
        }
    }
}

#[derive(Debug)]
pub struct TextureBufferInfo {
    pub buffer: MaybeOwnedBuffer,
    pub copy_area: PixelRegion,
}

#[derive(Debug)]
pub struct TextureTarget {
    pub(crate) cached_view: Option<(
        crate::gpu_runtime::ResourceId,
        crate::gpu_resources::TextureView,
    )>,
    pub size: wgpu::Extent3d,
    pub texture: crate::gpu_resources::Texture,
    pub format: wgpu::TextureFormat,
    pub buffer: Option<TextureBufferInfo>,
}

#[derive(Debug)]
pub struct TextureTargetFrame(crate::gpu_resources::TextureView);

impl RenderTargetFrame for TextureTargetFrame {
    fn view(&self) -> &crate::gpu_resources::TextureView {
        &self.0
    }

    fn into_view(self) -> crate::gpu_resources::TextureView {
        self.0
    }
}

impl TextureTarget {
    /// Use an already tracked texture as a render-only target, preserving its
    /// resource identity and owner without allocating replacement storage.
    pub fn from_texture(texture: crate::gpu_resources::Texture) -> Self {
        Self {
            cached_view: None,
            size: texture.size(),
            format: texture.format(),
            texture,
            buffer: None,
        }
    }

    pub fn new(device: &crate::gpu_resources::Device, size: (u32, u32)) -> Result<Self, Error> {
        Self::with_readback(device, size, true)
    }

    /// Render-only target: neither allocates a mapping buffer nor copies into it.
    pub fn new_without_readback(
        device: &crate::gpu_resources::Device,
        size: (u32, u32),
    ) -> Result<Self, Error> {
        Self::with_readback(device, size, false)
    }

    fn with_readback(
        device: &crate::gpu_resources::Device,
        size: (u32, u32),
        readback: bool,
    ) -> Result<Self, Error> {
        if size.0 > device.limits().max_texture_dimension_2d
            || size.1 > device.limits().max_texture_dimension_2d
            || size.0 < 1
            || size.1 < 1
        {
            return Err(Error::new(
                RenderFailureKind::Validation,
                format!(
                    "Texture target cannot be smaller than 1 or larger than {}px on either dimension (requested {} x {})",
                    device.limits().max_texture_dimension_2d,
                    size.0,
                    size.1
                ),
            ));
        }
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let buffer_dimensions = BufferDimensions::new(size.0 as usize, size.1 as usize, format);
        let size = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };
        let texture_label = create_debug_label!("Render target texture");
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: texture_label.as_deref(),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            view_formats: &[format],
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
        })?;
        let buffer_label = create_debug_label!("Render target buffer");
        let buffer = readback
            .then(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: buffer_label.as_deref(),
                    size: (buffer_dimensions.padded_bytes_per_row as u64
                        * buffer_dimensions.height as u64),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .transpose()?;
        Ok(Self {
            cached_view: None,
            size,
            texture,
            format,
            buffer: buffer.map(|buffer| TextureBufferInfo {
                buffer: MaybeOwnedBuffer::Owned(buffer, buffer_dimensions),
                copy_area: PixelRegion::for_whole_size(size.width, size.height),
            }),
        })
    }

    pub fn get_texture(&self) -> crate::gpu_resources::Texture {
        self.texture.clone()
    }

    pub fn take_buffer(self) -> Option<TextureBufferInfo> {
        self.buffer
    }
}

impl RenderTarget for TextureTarget {
    type Frame = TextureTargetFrame;

    fn resize(
        &mut self,
        device: &crate::gpu_resources::Device,
        width: u32,
        height: u32,
    ) -> Result<(), Error> {
        *self = Self::with_readback(device, (width, height), self.buffer.is_some())?;

        Ok(())
    }

    fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    fn width(&self) -> u32 {
        self.size.width
    }

    fn height(&self) -> u32 {
        self.size.height
    }

    fn get_next_texture(&mut self) -> Result<Option<Self::Frame>, Error> {
        self.texture.lease().check_health()?;
        let id = self.texture.lease().id();
        if self
            .cached_view
            .as_ref()
            .is_none_or(|(source, _)| *source != id)
        {
            self.cached_view = Some((id, self.texture.create_view(&Default::default())?));
        }
        Ok(self
            .cached_view
            .as_ref()
            .map(|(_, view)| TextureTargetFrame(view.clone())))
    }

    #[instrument(level = "debug", skip_all)]
    fn submit<I: IntoIterator<Item = crate::gpu_resources::CommandBuffer>>(
        &self,
        device: &crate::gpu_resources::Device,
        queue: &crate::gpu_resources::Queue,
        health: &RendererHealth,
        command_buffers: I,
        _frame: Self::Frame,
    ) -> Result<wgpu::SubmissionIndex, Error> {
        health.check()?;
        let index = if let Some(TextureBufferInfo { buffer, copy_area }) = &self.buffer {
            let label = create_debug_label!("Render target transfer encoder");
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: label.as_deref(),
            })?;
            let (buffer, dimensions) = buffer.inner();
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: copy_area.x_min,
                        y: copy_area.y_min,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(dimensions.padded_bytes_per_row),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: copy_area.width(),
                    height: copy_area.height(),
                    depth_or_array_layers: 1,
                },
            );
            let buffer = encoder.finish()?;
            ruffle_render::profile::work(|w| {
                w.copy_bytes += u64::from(copy_area.width()) * u64::from(copy_area.height()) * 4
            });
            let _timer = ruffle_render::profile::native_time(
                ruffle_render::profile::NativePhase::QueueSubmit,
            );
            queue.submit(command_buffers.into_iter().chain(Some(buffer)))?
        } else {
            let _timer = ruffle_render::profile::native_time(
                ruffle_render::profile::NativePhase::QueueSubmit,
            );
            queue.submit(command_buffers)?
        };
        health.check()?;
        Ok(index)
    }
}
