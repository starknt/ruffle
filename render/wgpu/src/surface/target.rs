use crate::backend::RenderTargetMode;
use crate::buffer_pool::{AlwaysCompatible, PoolEntry, TexturePool};
use crate::descriptors::Descriptors;
use crate::globals::Globals;
use crate::utils::run_copy_pipeline;
use std::cell::{OnceCell, RefCell};
use std::sync::Arc;

#[derive(Debug)]
pub struct ResolveBuffer {
    texture: PoolOrArcTexture,
}

impl ResolveBuffer {
    pub fn new(
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        pool: &mut TexturePool,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let texture = pool.get_texture(descriptors, size, usage, format, 1)?;
        Ok(Self {
            texture: PoolOrArcTexture::Pool(texture),
        })
    }

    pub fn from_view(
        view: crate::gpu_resources::TextureView,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        Ok(Self {
            texture: PoolOrArcTexture::Manual((view.texture_handle()?, view)),
        })
    }

    pub fn new_manual(
        texture: crate::gpu_resources::Texture,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        Ok(Self {
            texture: PoolOrArcTexture::Manual((
                texture.clone(),
                texture.create_view(&Default::default())?,
            )),
        })
    }

    pub fn view(&self) -> &crate::gpu_resources::TextureView {
        self.texture.view()
    }

    pub fn texture(&self) -> &crate::gpu_resources::Texture {
        self.texture.texture()
    }

    pub fn take_texture(self) -> PoolOrArcTexture {
        self.texture
    }
}

#[derive(Debug)]
pub struct FrameBuffer {
    texture: PoolOrArcTexture,
    size: wgpu::Extent3d,
}

#[derive(Debug, PartialEq)]
/// Holds either a `PoolEntry` texture, or an `Arc`-wrapped texture.
/// This is used to select between using a texture pool for our framebuffer/resolve-buffer
/// (when rendering to the main screen), or rendering to a non-pooled `Texture`
/// (when doing an offscreen render to a BitmapData texture)
pub enum PoolOrArcTexture {
    Shared(Arc<PoolOrArcTexture>),
    Pool(
        PoolEntry<
            (
                crate::gpu_resources::Texture,
                crate::gpu_resources::TextureView,
            ),
            AlwaysCompatible,
        >,
    ),
    Manual(
        (
            crate::gpu_resources::Texture,
            crate::gpu_resources::TextureView,
        ),
    ),
}

impl PoolOrArcTexture {
    pub fn texture(&self) -> &crate::gpu_resources::Texture {
        match self {
            PoolOrArcTexture::Shared(texture) => texture.texture(),
            PoolOrArcTexture::Pool(texture) => &texture.0,
            PoolOrArcTexture::Manual(texture) => &texture.0,
        }
    }
    pub fn view(&self) -> &crate::gpu_resources::TextureView {
        match self {
            PoolOrArcTexture::Shared(texture) => texture.view(),
            PoolOrArcTexture::Pool(texture) => &texture.1,
            PoolOrArcTexture::Manual(texture) => &texture.1,
        }
    }
}

impl FrameBuffer {
    pub fn new(
        descriptors: &Descriptors,
        sample_count: u32,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        pool: &mut TexturePool,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let texture = pool.get_texture(descriptors, size, usage, format, sample_count)?;

        Ok(Self {
            texture: PoolOrArcTexture::Pool(texture),
            size,
        })
    }

    pub fn from_view(
        view: crate::gpu_resources::TextureView,
        size: wgpu::Extent3d,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        Ok(Self {
            texture: PoolOrArcTexture::Manual((view.texture_handle()?, view)),
            size,
        })
    }

    pub fn new_manual(
        texture: crate::gpu_resources::Texture,
        size: wgpu::Extent3d,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        Ok(Self {
            texture: PoolOrArcTexture::Manual((
                texture.clone(),
                texture.create_view(&Default::default())?,
            )),
            size,
        })
    }

    pub fn view(&self) -> &crate::gpu_resources::TextureView {
        self.texture.view()
    }

    pub fn texture(&self) -> &crate::gpu_resources::Texture {
        self.texture.texture()
    }

    pub fn take_texture(self) -> PoolOrArcTexture {
        self.texture
    }

    pub fn size(&self) -> wgpu::Extent3d {
        self.size
    }
}

#[derive(Debug)]
pub struct BlendBuffer {
    texture: PoolEntry<
        (
            crate::gpu_resources::Texture,
            crate::gpu_resources::TextureView,
        ),
        AlwaysCompatible,
    >,
}

impl BlendBuffer {
    pub fn new(
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        pool: &mut TexturePool,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let texture = pool.get_texture(descriptors, size, usage, format, 1)?;

        Ok(Self { texture })
    }

    pub fn view(&self) -> &crate::gpu_resources::TextureView {
        &self.texture.1
    }

    pub fn texture(&self) -> &crate::gpu_resources::Texture {
        &self.texture.0
    }
}

#[derive(Debug)]
pub struct StencilBuffer {
    texture: PoolEntry<
        (
            crate::gpu_resources::Texture,
            crate::gpu_resources::TextureView,
        ),
        AlwaysCompatible,
    >,
}

impl StencilBuffer {
    pub fn new(
        descriptors: &Descriptors,
        msaa_sample_count: u32,
        size: wgpu::Extent3d,
        pool: &mut TexturePool,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let texture = pool.get_texture(
            descriptors,
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            wgpu::TextureFormat::Stencil8,
            msaa_sample_count,
        )?;

        Ok(Self { texture })
    }

    pub fn view(&self) -> &crate::gpu_resources::TextureView {
        &self.texture.1
    }
}

pub struct CommandTarget {
    frame_buffer: FrameBuffer,
    blend_buffer: OnceCell<BlendBuffer>,
    resolve_buffer: Option<ResolveBuffer>,
    depth: OnceCell<StencilBuffer>,
    globals: Arc<Globals>,
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
    sample_count: u32,
    color_needs_clear: OnceCell<bool>,
    render_target_mode: RenderTargetMode,
    resolved_changes: RefCell<Vec<ruffle_render::region::RenderRect>>,
}

impl CommandTarget {
    pub fn new(
        descriptors: &Descriptors,
        pool: &mut TexturePool,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        sample_count: u32,
        render_target_mode: RenderTargetMode,
        encoder: &mut crate::gpu_resources::CommandEncoder,
    ) -> Result<Self, crate::gpu_runtime::RenderFailure> {
        let globals = pool.get_globals(descriptors, size.width, size.height)?;

        let mut make_pooled_frame_buffer = || {
            FrameBuffer::new(
                descriptors,
                sample_count,
                size,
                format,
                if sample_count > 1 {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                } else {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::TEXTURE_BINDING
                },
                pool,
            )
        };

        let (frame_buffer, resolve_buffer) =
            if let RenderTargetMode::ExistingWithView(view, _) = &render_target_mode {
                if sample_count > 1 {
                    (
                        make_pooled_frame_buffer()?,
                        Some(ResolveBuffer::from_view(view.clone())?),
                    )
                } else {
                    (FrameBuffer::from_view(view.clone(), size)?, None)
                }
            } else if let RenderTargetMode::ExistingWithColor(texture, _) = &render_target_mode {
                if sample_count > 1 {
                    (
                        make_pooled_frame_buffer()?,
                        Some(ResolveBuffer::new_manual(texture.clone())?),
                    )
                } else {
                    (
                        FrameBuffer::new_manual(texture.clone(), texture.size())?,
                        None,
                    )
                }
            } else if sample_count > 1 {
                (
                    make_pooled_frame_buffer()?,
                    Some(ResolveBuffer::new(
                        descriptors,
                        size,
                        format,
                        wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::RENDER_ATTACHMENT,
                        pool,
                    )?),
                )
            } else {
                (make_pooled_frame_buffer()?, None)
            };

        if let RenderTargetMode::FreshWithTexture(texture) = &render_target_mode {
            if let Some(resolve_buffer) = &resolve_buffer {
                ruffle_render::profile::work(|w| {
                    w.copy_bytes += u64::from(size.width) * u64::from(size.height) * 4
                });
                encoder.copy_texture_to_texture(
                    texture.as_image_copy(),
                    resolve_buffer.texture().as_image_copy(),
                    size,
                );
            }

            if sample_count > 1 {
                // Both our frame buffer and resolve buffer need to start out
                // in the same state, so copy our existing texture to the freshly
                // allocated frame buffer. We cannot use `copy_texture_to_texture`,
                // since the sample counts are different.
                run_copy_pipeline(
                    descriptors,
                    format,
                    frame_buffer.texture.view(),
                    &texture.create_view(&Default::default())?,
                    sample_count,
                    encoder,
                )?;
            } else {
                ruffle_render::profile::work(|w| {
                    w.copy_bytes += u64::from(size.width) * u64::from(size.height) * 4
                });
                encoder.copy_texture_to_texture(
                    texture.as_image_copy(),
                    frame_buffer.texture().as_image_copy(),
                    size,
                );
            }
        }

        Ok(Self {
            frame_buffer,
            blend_buffer: OnceCell::new(),
            resolve_buffer,
            depth: OnceCell::new(),
            globals,
            size,
            format,
            sample_count,
            color_needs_clear: OnceCell::new(),
            render_target_mode,
            resolved_changes: RefCell::default(),
        })
    }

    pub fn width(&self) -> u32 {
        self.size.width
    }

    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn ensure_cleared(&self, encoder: &mut crate::gpu_resources::CommandEncoder) {
        if self.color_needs_clear.get().is_some() {
            return;
        }
        // If we aren't clearing with a color (eg a texture instead)
        // the there's no point in creating a new render pass that does nothing.
        if self.render_target_mode.color().is_some() {
            ruffle_render::profile::work(|w| {
                w.render_passes += 1;
                w.pass_pixels += u64::from(self.width()) * u64::from(self.height());
            });
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: create_debug_label!("Clearing command target").as_deref(),
                color_attachments: &[self.color_attachments()],
                ..Default::default()
            });
        }
    }

    pub fn take_color_texture(self) -> PoolOrArcTexture {
        self.resolve_buffer
            .map(|b| b.take_texture())
            .unwrap_or_else(|| self.frame_buffer.take_texture())
    }

    pub fn globals(&self) -> &Globals {
        &self.globals
    }

    pub fn can_composite_resolved(&self) -> bool {
        self.resolve_buffer.as_ref().is_some_and(|buffer| {
            buffer
                .texture()
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        })
    }

    /// Pointwise compositing writes the same value to every MSAA sample. Keep
    /// those writes in the resolve image until geometry needs individual samples.
    pub fn resolved_attachment(
        &self,
        region: ruffle_render::region::RenderRect,
    ) -> Option<wgpu::RenderPassColorAttachment<'_>> {
        self.resolved_changes.borrow_mut().push(region);
        Some(wgpu::RenderPassColorAttachment {
            view: self.color_view(),
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })
    }

    pub fn restore_multisample_regions(
        &self,
        descriptors: &Descriptors,
        encoder: &mut crate::gpu_resources::CommandEncoder,
    ) -> Result<(), crate::gpu_runtime::RenderFailure> {
        let mut regions = self.resolved_changes.borrow_mut();
        if regions.is_empty() {
            return Ok(());
        }
        // Never restore a bounding union: gaps may still contain distinct AA
        // samples. Each changed rectangle was fully overwritten by a blend quad.
        crate::utils::run_copy_regions_pipeline(
            descriptors,
            self.format,
            self.frame_buffer.view(),
            self.color_view(),
            self.sample_count,
            encoder,
            Some(&regions),
        )?;
        regions.clear();
        Ok(())
    }

    pub fn color_attachments(&self) -> Option<wgpu::RenderPassColorAttachment<'_>> {
        let mut load = wgpu::LoadOp::Load;
        if self.color_needs_clear.set(false).is_ok()
            && let Some(clear_color) = self.render_target_mode.color()
        {
            load = wgpu::LoadOp::Clear(clear_color);
        }
        Some(wgpu::RenderPassColorAttachment {
            view: self.frame_buffer.view(),
            resolve_target: self.resolve_buffer.as_ref().map(|b| b.view().raw()),
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn stencil_attachment(
        &self,
        descriptors: &Descriptors,
        pool: &mut TexturePool,
    ) -> Result<Option<wgpu::RenderPassDepthStencilAttachment<'_>>, crate::gpu_runtime::RenderFailure>
    {
        let new_buffer = self.depth.get().is_none();
        let stencil = crate::gpu_resources::try_cell(&self.depth, || {
            StencilBuffer::new(descriptors, self.sample_count, self.size, pool)
        })?;
        Ok(Some(wgpu::RenderPassDepthStencilAttachment {
            view: stencil.view(),
            depth_ops: None,
            stencil_ops: Some(wgpu::Operations {
                load: if new_buffer {
                    wgpu::LoadOp::Clear(0)
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            }),
        }))
    }

    pub fn update_blend_buffer(
        &self,
        descriptors: &Descriptors,
        pool: &mut TexturePool,
        encoder: &mut crate::gpu_resources::CommandEncoder,
    ) -> Result<&BlendBuffer, crate::gpu_runtime::RenderFailure> {
        let blend_buffer = crate::gpu_resources::try_cell(&self.blend_buffer, || {
            BlendBuffer::new(
                descriptors,
                self.size,
                self.format,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                pool,
            )
        })?;
        self.ensure_cleared(encoder);
        ruffle_render::profile::work(|w| {
            w.copy_bytes += u64::from(self.size.width) * u64::from(self.size.height) * 4
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self
                    .resolve_buffer
                    .as_ref()
                    .map(|b| b.texture())
                    .unwrap_or_else(|| self.frame_buffer.texture()),
                mip_level: 0,
                origin: Default::default(),
                aspect: Default::default(),
            },
            wgpu::TexelCopyTextureInfo {
                texture: blend_buffer.texture(),
                mip_level: 0,
                origin: Default::default(),
                aspect: Default::default(),
            },
            self.frame_buffer.size(),
        );
        Ok(blend_buffer)
    }

    pub fn copy_blend_region(
        &self,
        descriptors: &Descriptors,
        pool: &mut TexturePool,
        encoder: &mut crate::gpu_resources::CommandEncoder,
        region: ruffle_render::region::RenderRect,
    ) -> Result<BlendBuffer, crate::gpu_runtime::RenderFailure> {
        let size = wgpu::Extent3d {
            width: region.width,
            height: region.height,
            depth_or_array_layers: 1,
        };
        let buffer = BlendBuffer::new(
            descriptors,
            size,
            self.format,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            pool,
        )?;
        self.ensure_cleared(encoder);
        ruffle_render::profile::work(|w| {
            w.copy_bytes += u64::from(region.width) * u64::from(region.height) * 4
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.color_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: region.x,
                    y: region.y,
                    z: 0,
                },
                aspect: Default::default(),
            },
            wgpu::TexelCopyTextureInfo {
                texture: buffer.texture(),
                mip_level: 0,
                origin: Default::default(),
                aspect: Default::default(),
            },
            size,
        );
        Ok(buffer)
    }

    pub fn color_view(&self) -> &crate::gpu_resources::TextureView {
        self.resolve_buffer
            .as_ref()
            .map(|b| b.view())
            .unwrap_or_else(|| self.frame_buffer.view())
    }

    pub fn color_texture(&self) -> &crate::gpu_resources::Texture {
        self.resolve_buffer
            .as_ref()
            .map(|b| b.texture())
            .unwrap_or_else(|| self.frame_buffer.texture())
    }
}
