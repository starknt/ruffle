//! Main-thread ownership pairs immutable, Send geometry with confined bitmap
//! handles. Workers can receive the geometry without acquiring renderer handles.
use crate::gpu_runtime::{RenderFailure, ResourceOwner};
use crate::{backend::WgpuRenderBackend, mesh::Mesh, target::RenderTarget};
use ruffle_render::{
    backend::{RenderBackend, ShapeHandleImpl},
    bitmap::{BitmapHandle, BitmapSize, BitmapSource},
    shape_utils::OwnedDistilledShape,
};
use std::{
    cell::{OnceCell, RefCell},
    collections::HashMap,
    sync::Arc,
};

#[derive(Debug, Default)]
pub struct FrozenBitmapSource(pub HashMap<u16, (BitmapSize, Option<BitmapHandle>)>);
impl BitmapSource for FrozenBitmapSource {
    fn bitmap_size(&self, id: u16) -> Option<BitmapSize> {
        self.0.get(&id).map(|(size, _)| *size)
    }
    fn bitmap_handle(&self, id: u16, _: &mut dyn RenderBackend) -> Option<BitmapHandle> {
        self.0.get(&id).and_then(|(_, handle)| handle.clone())
    }
}
#[derive(Debug)]
pub struct ShapeRecipe {
    pub static_definition: std::cell::Cell<bool>,
    pub owner: ResourceOwner,
    pub device_generation: u64,
    pub mesh: OnceCell<Arc<Mesh>>,
    pub geometry: RefCell<Option<Arc<OwnedDistilledShape>>>,
    pub bitmaps: FrozenBitmapSource,
    pub scale: f32,
    pub(crate) _lifetime: crate::render_snapshot::ResourceLifetime,
}
impl ShapeRecipe {
    pub fn publish<T: RenderTarget>(
        &self,
        backend: &mut WgpuRenderBackend<T>,
        prepared: crate::prepare::mesh::PreparedMesh,
    ) -> Result<(), RenderFailure> {
        if self.mesh.get().is_some() {
            return Ok(());
        }
        let id = self
            .geometry
            .borrow()
            .as_ref()
            .ok_or_else(|| {
                RenderFailure::new(
                    crate::gpu_runtime::RenderFailureKind::Internal,
                    "missing shape geometry at publication",
                )
            })?
            .id;
        let mesh = backend.register_prepared_shape(prepared, id, &self.bitmaps, self.owner)?;
        let _ = self.mesh.set(mesh);
        self.geometry.borrow_mut().take();
        Ok(())
    }
}
impl ShapeHandleImpl for ShapeRecipe {
    fn retained_bytes(&self) -> Option<usize> {
        self.mesh.get().and_then(|mesh| mesh.retained_bytes)
    }
    fn bounds(&self) -> Option<[f64; 4]> {
        self.mesh.get().and_then(|mesh| mesh.bounds)
    }
}
