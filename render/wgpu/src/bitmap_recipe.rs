//! Lazy native bitmap storage. Registration and CPU updates own their pixels;
//! allocation/upload happens only at an explicit GPU use on the player thread.
pub(crate) use crate::gpu_runtime::{OwnerKind, RenderFailure, RenderFailureKind, ResourceOwner};
use crate::{Texture, bitmap_residency::BitmapResidency, gpu_resources::Device};
use ruffle_render::bitmap::coverage::Coverage;
use ruffle_render::bitmap::{Bitmap, BitmapHandleImpl, PixelRegion};
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[derive(Clone, Debug)]
struct PendingUpload {
    pixels: Arc<Bitmap<'static>>,
    coverage: Coverage,
}
#[derive(Debug)]
struct State {
    version: u64,
    pending: Option<PendingUpload>,
    cache: Weak<BitmapResidency<BitmapRecipe>>,
}
#[derive(Debug)]
pub struct BitmapRecipe {
    pub owner: ResourceOwner,
    pub device_generation: u64,
    pub width: u32,
    pub height: u32,
    weak_self: Weak<Self>,
    state: Mutex<State>,
    resident: OnceLock<Texture>,
}
impl BitmapHandleImpl for BitmapRecipe {}
impl BitmapRecipe {
    pub fn new(pixels: Bitmap<'_>, device: &Device) -> Result<Arc<Self>, RenderFailure> {
        let owner = device.runtime().issue_owner(OwnerKind::BitmapData)?;
        let (width, height) = (pixels.width(), pixels.height());
        Ok(Arc::new_cyclic(|weak_self| Self {
            owner,
            device_generation: device.runtime().generation(),
            width,
            height,
            weak_self: weak_self.clone(),
            resident: OnceLock::new(),
            state: Mutex::new(State {
                version: 1,
                cache: Weak::new(),
                pending: Some(PendingUpload {
                    pixels: Arc::new(pixels.into_owned()),
                    coverage: Coverage::new(
                        PixelRegion::for_whole_size(width, height),
                        width,
                        height,
                    ),
                }),
            }),
        }))
    }
    pub fn pending_upload(&self) -> Option<(Arc<Bitmap<'static>>, Coverage)> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
            .as_ref()
            .map(|pending| (pending.pixels.clone(), pending.coverage.clone()))
    }
    pub fn version(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .version
    }
    pub fn is_resident(&self) -> bool {
        self.resident.get().is_some()
    }
    pub fn retain_in(&self, cache: &Arc<BitmapResidency<Self>>) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cache = Arc::downgrade(cache);
        if self.is_resident() {
            self.touch_cache();
        }
    }
    fn touch_cache(&self) {
        let cache = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cache
            .upgrade();
        if let Some(cache) = cache
            && let Some(this) = self.weak_self.upgrade()
        {
            if let Some(resident) = self.resident.get() {
                resident.texture.lease().set_reconstructible(true);
            }
            cache.retain(&this, u64::from(self.width) * u64::from(self.height) * 4);
            ruffle_render::profile::record_bitmap_residency(cache.stats());
        }
    }
    pub fn update(&self, pixels: Bitmap<'_>, region: PixelRegion) -> Result<(), RenderFailure> {
        if pixels.width() != self.width
            || pixels.height() != self.height
            || region.x_max > self.width
            || region.y_max > self.height
        {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "bitmap update exceeds its recipe dimensions",
            ));
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.version = state.version.checked_add(1).ok_or_else(|| {
            RenderFailure::new(RenderFailureKind::Internal, "bitmap version exhausted")
        })?;
        if let Some(pending) = &mut state.pending {
            if !Arc::make_mut(&mut pending.pixels).copy_region_from(&pixels, region) {
                return Err(RenderFailure::new(
                    RenderFailureKind::Validation,
                    "invalid bitmap patch",
                ));
            }
            pending.coverage.add(region);
        } else {
            state.pending = Some(PendingUpload {
                pixels: Arc::new(pixels.into_owned()),
                coverage: Coverage::new(region, self.width, self.height),
            });
        }
        Ok(())
    }
    /// Called after an ordered GPU write is encoded. Native/Stage3D handles
    /// without a recipe remain opaque graph barriers.
    pub fn record_gpu_write(&self) -> Result<(), RenderFailure> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.pending.is_some() {
            return Err(RenderFailure::new(
                RenderFailureKind::Internal,
                "GPU destination has unresolved CPU writes",
            ));
        }
        state.version = state.version.checked_add(1).ok_or_else(|| {
            RenderFailure::new(RenderFailureKind::Internal, "bitmap version exhausted")
        })?;
        Ok(())
    }
    pub fn resolve(&self, device: &Device) -> Result<&Texture, RenderFailure> {
        device.runtime().check_health()?;
        if self.device_generation != device.runtime().generation() {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "bitmap recipe belongs to another device",
            ));
        }
        let device = device.with_owner(self.owner);
        let texture = crate::gpu_resources::try_sync_cell(&self.resident, || {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Bitmap recipe"),
                size: wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
            })?;
            Ok::<_, RenderFailure>(Texture {
                _allocation: ruffle_render::profile::GpuBitmap::new(
                    u64::from(self.width) * u64::from(self.height) * 4,
                ),
                view: texture.create_view(&Default::default())?,
                texture,
                repeating_linear: Default::default(),
                repeating_nearest: Default::default(),
                clamped_linear: Default::default(),
                clamped_nearest: Default::default(),
                copy_count: Default::default(),
                version: Default::default(),
            })
        })?;
        let (version, pending) = {
            let state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (state.version, state.pending.clone())
        };
        if let Some(pending) = pending {
            let _upload = ruffle_render::profile::time(ruffle_render::profile::Phase::Upload);
            pending
                .coverage
                .for_each(|region| -> Result<(), RenderFailure> {
                    device.queue().write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: region.x_min,
                                y: region.y_min,
                                z: 0,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        pending.pixels.data(),
                        wgpu::TexelCopyBufferLayout {
                            offset: (u64::from(region.y_min) * u64::from(self.width)
                                + u64::from(region.x_min))
                                * 4,
                            bytes_per_row: Some(self.width * 4),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width: region.width(),
                            height: region.height(),
                            depth_or_array_layers: 1,
                        },
                    )?;
                    ruffle_render::profile::uploaded(
                        u64::from(region.width()) * u64::from(region.height()) * 4,
                    );
                    Ok(())
                })?;
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.version == version {
                state.pending = None;
            }
        }
        self.touch_cache();
        Ok(texture)
    }
}
