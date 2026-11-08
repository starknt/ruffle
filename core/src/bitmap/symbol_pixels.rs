//! Embedded bitmap frames keep their original lossless/JPEG source. An AS3
//! BitmapData can be displayed without also materializing a private CPU copy.
//! Only the display-list read path shares a texture; all writable GPU handles
//! and all modified CPU buffers are private to the BitmapData.
use super::bitmap_data::Color;
use crate::character::CompressedBitmap;
use ruffle_render::{
    backend::RenderBackend,
    bitmap::{BitmapHandle, BitmapHandleImpl},
    profile::BitmapAllocation,
};
use std::{
    cell::{OnceCell, RefCell},
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::Weak,
};

#[derive(Debug)]
pub struct SymbolImage {
    pub data: CompressedBitmap,
    view: RefCell<Option<Weak<dyn BitmapHandleImpl>>>,
}
impl SymbolImage {
    pub fn new(data: CompressedBitmap) -> Self {
        Self {
            data,
            view: Default::default(),
        }
    }
    fn view(
        &self,
        renderer: &mut dyn RenderBackend,
    ) -> Result<BitmapHandle, ruffle_render::error::Error> {
        if let Some(h) = self.view.borrow().as_ref().and_then(Weak::upgrade) {
            return Ok(BitmapHandle(h));
        }
        let h = renderer.register_bitmap(self.data.decode()?)?;
        *self.view.borrow_mut() = Some(std::sync::Arc::downgrade(&h.0));
        Ok(h)
    }
}
pub struct Pixels {
    symbol: Option<Rc<SymbolImage>>,
    view: Option<BitmapHandle>,
    data: OnceCell<(Vec<Color>, BitmapAllocation)>,
}
impl Pixels {
    pub fn owned(p: Vec<Color>) -> Self {
        let a = BitmapAllocation::new(p.capacity() * size_of::<Color>());
        Self {
            symbol: None,
            view: None,
            data: OnceCell::from((p, a)),
        }
    }
    pub fn symbol(s: Rc<SymbolImage>) -> Self {
        Self {
            symbol: Some(s),
            view: None,
            data: OnceCell::new(),
        }
    }
    pub fn view(&mut self, renderer: &mut dyn RenderBackend) -> Option<BitmapHandle> {
        let s = self.symbol.as_ref()?;
        if self.view.is_none() {
            self.view = s.view(renderer).ok()
        }
        self.view.clone()
    }
    fn materialize(&self) -> &(Vec<Color>, BitmapAllocation) {
        self.data.get_or_init(|| {
            let decoded = self
                .symbol
                .as_ref()
                .unwrap()
                .data
                .decode()
                .expect("Failed to decode BitmapData");
            let p: Vec<_> = decoded.as_colors().map(Color::from).collect();
            let a = BitmapAllocation::new(p.capacity() * size_of::<Color>());
            (p, a)
        })
    }
}
impl Clone for Pixels {
    fn clone(&self) -> Self {
        if let Some(s) = &self.symbol {
            Self::symbol(s.clone())
        } else {
            Self::owned(self.deref().clone())
        }
    }
}
impl Deref for Pixels {
    type Target = Vec<Color>;
    fn deref(&self) -> &Self::Target {
        &self.materialize().0
    }
}
impl DerefMut for Pixels {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.materialize();
        self.symbol = None;
        self.view = None;
        &mut self.data.get_mut().unwrap().0
    }
}
