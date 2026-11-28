#![deny(clippy::unwrap_used)]

use std::{
    any::Any,
    cell::{RefCell, RefMut},
    rc::Rc,
};

pub mod backend;
pub mod error;
pub mod frame;
pub mod null;

/// A movie or NetStream owns its decoder and last texture through this shared handle.
#[derive(Clone, Debug)]
pub struct VideoStreamHandle(Rc<RefCell<dyn Any>>);

impl VideoStreamHandle {
    pub fn new<T: Any>(stream: T) -> Self {
        Self(Rc::new(RefCell::new(stream)))
    }

    pub fn borrow_mut<T: Any>(&self) -> Result<RefMut<'_, T>, error::Error> {
        RefMut::filter_map(self.0.borrow_mut(), |stream| stream.downcast_mut::<T>())
            .map_err(|_| error::Error::VideoStreamIsNotRegistered)
    }
}
