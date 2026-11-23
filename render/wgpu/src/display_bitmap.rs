//! Optional display-list sampling. Writable handles and readback resolve through
//! the original bitmap; only a stage-owned surface can request a derived view.
pub use crate::render_snapshot::{ResourceKey, ResourceVersion};
pub use crate::resource_scheduler::{CacheTicket, Reclaim, ReclaimClass, ResourceScheduler};
use crate::{bitmap_recipe::BitmapRecipe, gpu_resources, gpu_runtime::RenderFailure};
pub use ruffle_render::backend::BitmapUsage;
use ruffle_render::bitmap::BitmapHandle;
use std::{
    any::Any,
    sync::{Arc, Weak},
};

#[derive(Clone, Debug)]
pub struct Source {
    recipe: Weak<BitmapRecipe>,
    pub version: ResourceVersion,
    pub size: [u32; 2],
    pub usage: BitmapUsage,
    pub detail: Option<BitmapDetail>,
}

/// Bounded samples of the original pixels taken while they are CPU-resident.
#[derive(Clone, Copy, Debug)]
pub struct BitmapDetail {
    pub transparent: u8,
    pub edges: u8,
}
impl Source {
    pub(crate) fn for_handle(handle: &BitmapHandle) -> Option<Self> {
        let recipe = (handle.0.as_ref() as &dyn Any).downcast_ref::<BitmapRecipe>()?;
        if recipe.usage == BitmapUsage::Excluded {
            return None;
        }
        Some(Self {
            recipe: recipe.weak_self.clone(),
            version: ResourceVersion {
                key: ResourceKey::Recipe {
                    device: recipe.device_generation,
                    owner: recipe.owner,
                },
                version: recipe.version(),
            },
            size: [recipe.width, recipe.height],
            usage: recipe.usage,
            detail: recipe.detail,
        })
    }
    pub fn current(&self) -> bool {
        self.recipe
            .upgrade()
            .is_some_and(|r| r.version() == self.version.version)
    }
    pub fn resolve(
        &self,
        device: &gpu_resources::Device,
    ) -> Result<Option<gpu_resources::Texture>, RenderFailure> {
        let Some(r) = self
            .recipe
            .upgrade()
            .filter(|r| r.version() == self.version.version)
        else {
            return Ok(None);
        };
        Ok(Some(r.resolve(device)?.texture.clone()))
    }
    pub(crate) fn refresh(&self) -> Option<Self> {
        let r = self.recipe.upgrade()?;
        let mut source = self.clone();
        source.version.version = r.version();
        Some(source)
    }
}

#[derive(Clone, Debug)]
pub struct DisplaySample {
    pub texture: Arc<crate::Texture>,
    /// Bounds of the original full image in the backing texture, in normalized coordinates.
    pub uv: [f32; 4],
}
impl DisplaySample {
    pub fn full(texture: gpu_resources::Texture) -> Result<Self, RenderFailure> {
        let view = texture.create_view(&Default::default())?;
        Ok(Self {
            texture: Arc::new(crate::Texture {
                _allocation: ruffle_render::profile::GpuBitmap::new(
                    u64::from(texture.width()) * u64::from(texture.height()) * 4,
                ),
                texture,
                view,
                batch_source: Default::default(),
                repeating_linear: Default::default(),
                repeating_nearest: Default::default(),
                clamped_linear: Default::default(),
                clamped_nearest: Default::default(),
                pending_read: Default::default(),
                _lifetime: None,
                version: Default::default(),
            }),
            uv: [0., 0., 1., 1.],
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sampling {
    Direct,
    ClampedFill,
    RepeatingFill,
}
pub trait DisplayBitmapEnhancer: std::fmt::Debug + Send + Sync {
    fn enabled(&self) -> bool;
    fn revision(&self) -> u64;
    fn sample(
        &self,
        source: Source,
        scale: f32,
        sampling: Sampling,
        encoder: &mut gpu_resources::CommandEncoder,
    ) -> Result<Option<DisplaySample>, RenderFailure>;
    fn finish_frame(
        &self,
        encoder: &mut gpu_resources::CommandEncoder,
    ) -> Result<(), RenderFailure>;
    fn submitted(&self, queue: &wgpu::Queue);
}
