//! Window-owned swapchain recovery. Movie and egui retain their device and
//! pipelines across a window change, even if the new output format changes.
use anyhow::{Context, Result};
use ruffle_render_wgpu::{
    descriptors::Descriptors, gpu_resources as gpu, target::PresentationSurface,
};
use std::sync::Arc;
use wgpu::{TextureFormat, TextureUsages};
use winit::{dpi::PhysicalSize, window::Window};

pub(super) struct DesktopSurface {
    target: PresentationSurface,
    render_format: TextureFormat,
    conversion: Option<FormatConversion>,
}

struct FormatConversion {
    size: (u32, u32),
    render_view: gpu::TextureView,
    sample_view: gpu::TextureView,
    blitter: wgpu::util::TextureBlitter,
}

impl DesktopSurface {
    pub fn new(
        descriptors: &Descriptors,
        surface: wgpu::Surface<'static>,
        size: PhysicalSize<u32>,
    ) -> Result<Self> {
        let config = configure(descriptors, &surface, size, None)?;
        let render_format = config.format;
        Ok(Self {
            target: PresentationSurface::new(surface, config, &descriptors.device)?,
            render_format,
            conversion: None,
        })
    }

    pub fn render_format(&self) -> TextureFormat {
        self.render_format
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.target.resize(size.width, size.height);
    }

    pub fn acquire(
        &mut self,
        descriptors: &Descriptors,
        window: &Arc<Window>,
    ) -> Result<Option<wgpu::SurfaceTexture>> {
        descriptors.runtime.check_health()?;
        if self.target.needs_recreation() {
            let config = self.target.config();
            let size = PhysicalSize::new(config.width, config.height);
            let surface = descriptors
                .wgpu_instance
                .create_surface(Arc::clone(window))?;
            let config = configure(descriptors, &surface, size, Some(self.render_format))?;
            self.target = PresentationSurface::new(surface, config, &descriptors.device)?;
            self.conversion = None;
        }
        let config = self.target.config();
        if config.format != self.render_format
            && self.conversion.as_ref().map(|c| c.size) != Some((config.width, config.height))
        {
            self.conversion = Some(FormatConversion::new(
                &descriptors.device,
                config,
                self.render_format,
            )?);
        }
        Ok(self.target.acquire(&descriptors.device)?)
    }

    pub fn render_view<'a>(&'a self, window_view: &'a wgpu::TextureView) -> &'a wgpu::TextureView {
        self.conversion
            .as_ref()
            .map_or(window_view, |c| c.render_view.raw())
    }

    pub fn finish(
        &self,
        device: &gpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        window_view: &wgpu::TextureView,
    ) {
        if let Some(c) = &self.conversion {
            c.blitter.copy(device, encoder, &c.sample_view, window_view);
        }
    }

    pub fn view_leases(
        &self,
    ) -> impl Iterator<Item = ruffle_render_wgpu::gpu_runtime::ResourceLease> + '_ {
        self.conversion
            .iter()
            .flat_map(|c| [c.render_view.lease().clone(), c.sample_view.lease().clone()])
    }
}

fn configure(
    descriptors: &Descriptors,
    surface: &wgpu::Surface<'_>,
    size: PhysicalSize<u32>,
    preferred: Option<TextureFormat>,
) -> Result<wgpu::SurfaceConfiguration> {
    let caps = surface.get_capabilities(&descriptors.adapter);
    let mut config = surface
        .get_default_config(&descriptors.adapter, size.width.max(1), size.height.max(1))
        .context("window surface has no compatible format")?;
    config.format = preferred
        .into_iter()
        .chain([TextureFormat::Rgba8Unorm, TextureFormat::Bgra8Unorm])
        .find(|f| caps.formats.contains(f))
        .unwrap_or(config.format);
    config.present_mode = Default::default();
    config.alpha_mode = caps
        .alpha_modes
        .first()
        .copied()
        .context("window surface has no alpha mode")?;
    Ok(config)
}

impl FormatConversion {
    fn new(
        device: &gpu::Device,
        config: &wgpu::SurfaceConfiguration,
        render_format: TextureFormat,
    ) -> Result<Self> {
        // Present the already-composited SDR values. sRGB sampling paired with
        // an sRGB output cancels the automatic transfer functions in the blit.
        let sample_format = if config.format.is_srgb() {
            render_format.add_srgb_suffix()
        } else {
            render_format.remove_srgb_suffix()
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Desktop surface format conversion"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: render_format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            view_formats: if sample_format == render_format {
                &[]
            } else {
                std::slice::from_ref(&sample_format)
            },
        })?;
        Ok(Self {
            size: (config.width, config.height),
            render_view: texture.create_view(&Default::default())?,
            sample_view: texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(sample_format),
                ..Default::default()
            })?,
            blitter: wgpu::util::TextureBlitter::new(device, config.format),
        })
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use ruffle_render_wgpu::backend::{create_wgpu_instance, request_adapter_and_device};
    use winit::{event_loop::EventLoop, platform::x11::EventLoopBuilderExtX11};

    #[test]
    #[ignore = "requires a dedicated X11 display and Vulkan"]
    #[allow(deprecated)] // This test creates windows without running an application event loop.
    fn window_owner_recreates_a_surface_while_its_sibling_keeps_presenting() {
        let event_loop = EventLoop::builder()
            .with_x11()
            .with_any_thread(true)
            .build()
            .unwrap();
        let make_window = || {
            Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes().with_inner_size(PhysicalSize::new(64, 64)),
                    )
                    .unwrap(),
            )
        };
        let old = make_window();
        let sibling = make_window();
        let backends = wgpu::Backends::VULKAN;
        let instance = create_wgpu_instance(backends, Default::default(), None);
        let surface = instance.create_surface(old.clone()).unwrap();
        let other = instance.create_surface(sibling.clone()).unwrap();
        let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
            backends,
            &instance,
            Some(&surface),
            wgpu::PowerPreference::None,
        ))
        .unwrap();
        let descriptors = Descriptors::new(instance, adapter, device, queue).unwrap();
        let generation = descriptors.runtime.generation();
        let mut target = DesktopSurface::new(&descriptors, surface, old.inner_size()).unwrap();
        let mut second = DesktopSurface::new(&descriptors, other, sibling.inner_size()).unwrap();
        let present = |target: &mut DesktopSurface, window: &Arc<Window>| {
            let frame = target
                .acquire(&descriptors, window)
                .unwrap()
                .expect("window image");
            let output = frame.texture.create_view(&Default::default());
            let mut encoder = descriptors
                .device
                .raw()
                .create_command_encoder(&Default::default());
            drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.render_view(&output),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            }));
            target.finish(&descriptors.device, &mut encoder, &output);
            let leases = target.view_leases().collect::<Vec<_>>();
            descriptors
                .queue
                .submit_external([encoder.finish()], &leases)
                .unwrap();
            descriptors.queue.present(frame);
        };
        present(&mut target, &old);
        target.target.detach();
        let weak = Arc::downgrade(&old);
        drop(old);
        assert!(
            weak.upgrade().is_none(),
            "detaching must release the platform owner"
        );
        present(&mut second, &sibling);
        let replacement = make_window();
        // Exercise a renderer with a different attachment format on the same device.
        target.render_format = wgpu::TextureFormat::Rgba16Float;
        present(&mut target, &replacement);
        assert!(target.conversion.is_some());
        target.resize(PhysicalSize::new(48, 48));
        present(&mut target, &replacement);
        assert_eq!(target.conversion.as_ref().unwrap().size, (48, 48));
        assert_eq!(descriptors.runtime.generation(), generation);
        descriptors.runtime.check_health().unwrap();
    }
}
