use crate::descriptors::Descriptors;
use crate::globals::Globals;
use crate::idle_pool::Budget;
use fnv::FnvHashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::sync::Arc;

type Constructor<Type, Description> = Box<dyn Fn(&Descriptors, &Description) -> Type>;

#[derive(Debug)]
pub struct TexturePool {
    clock: u64,
    budget: Budget,
    pools: FnvHashMap<TextureKey, BufferPool<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible>>,
    globals_cache: FnvHashMap<GlobalsKey, (u64, Arc<Globals>)>,
}

impl TexturePool {
    pub fn new(budget: Budget) -> Self {
        Self {
            clock: 0,
            budget,
            pools: Default::default(),
            globals_cache: Default::default(),
        }
    }

    pub fn end_frame(&mut self) {
        self.clock += 1;
        if self.clock.is_multiple_of(16) {
            self.pools
                .retain(|_, p| Rc::weak_count(&p.owner) > 0 || self.budget.contains(p.owner.key));
            self.globals_cache
                .retain(|_, (used, _)| self.clock - *used <= 16);
        }
    }

    pub fn get_texture(
        &mut self,
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        usage: wgpu::TextureUsages,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> PoolEntry<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible> {
        let key = TextureKey {
            size,
            usage,
            format,
            sample_count,
        };
        let pool = self.pools.entry(key).or_insert_with(|| {
            let label = if cfg!(feature = "render_debug_labels") {
                use std::sync::atomic::{AtomicU32, Ordering};
                static ID_COUNT: AtomicU32 = AtomicU32::new(0);
                let id = ID_COUNT.fetch_add(1, Ordering::Relaxed);
                create_debug_label!("Pooled texture {}", id)
            } else {
                None
            };
            let bytes = size.width.div_ceil(format.block_dimensions().0) as u64
                * size.height.div_ceil(format.block_dimensions().1) as u64
                * u64::from(size.depth_or_array_layers)
                * u64::from(format.block_copy_size(None).unwrap_or(4))
                * u64::from(sample_count);
            BufferPool::new(
                self.budget.clone(),
                0,
                Box::new(move |_| bytes),
                Box::new(move |descriptors, _description| {
                    let texture = descriptors.device.create_texture(&wgpu::TextureDescriptor {
                        label: label.as_deref(),
                        size,
                        mip_level_count: 1,
                        sample_count,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        view_formats: &[format],
                        usage,
                    });
                    let view = texture.create_view(&Default::default());
                    (texture, view)
                }),
            )
        });
        pool.take(descriptors, AlwaysCompatible)
    }

    pub fn get_globals(
        &mut self,
        descriptors: &Descriptors,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Arc<Globals> {
        let slot = self
            .globals_cache
            .entry(GlobalsKey {
                viewport_width,
                viewport_height,
            })
            .or_insert_with(|| {
                (
                    self.clock,
                    Arc::new(Globals::new(
                        &descriptors.device,
                        &descriptors.bind_layouts.globals,
                        viewport_width,
                        viewport_height,
                    )),
                )
            });
        slot.0 = self.clock;
        slot.1.clone()
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct TextureKey {
    size: wgpu::Extent3d,
    usage: wgpu::TextureUsages,
    format: wgpu::TextureFormat,
    sample_count: u32,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct GlobalsKey {
    viewport_width: u32,
    viewport_height: u32,
}

pub trait BufferDescription: Clone + Debug + 'static {
    type Cost: Ord;

    /// If the potential buffer represented by this description (`self`)
    /// fits another existing buffer and its description (`other`),
    /// return the cost to use that buffer instead of making a new one.
    ///
    /// Cost is an arbitrary unit, but lower is better.
    /// None means that the other buffer cannot be used in place of this one.
    fn cost_to_use(&self, other: &Self) -> Option<Self::Cost>;
}

#[derive(Clone, Debug)]
pub struct AlwaysCompatible;

impl BufferDescription for AlwaysCompatible {
    type Cost = ();

    fn cost_to_use(&self, _other: &Self) -> Option<()> {
        Some(())
    }
}

// Stock keeps Ruffle's per-key vector and original lifetime, also when profiling.
struct Owner<T, D> {
    budget: Budget,
    key: u64,
    kind: usize,
    stock: Option<std::cell::RefCell<Vec<(T, D, u64)>>>,
}
impl<T, D> Drop for Owner<T, D> {
    fn drop(&mut self) {
        if let Some(v) = &self.stock {
            self.budget
                .stock_delta(self.kind, 0, v.borrow().iter().map(|e| e.2).sum())
        } else {
            self.budget.clear_pool(self.key);
        }
    }
}
pub struct BufferPool<T: 'static, D: BufferDescription> {
    owner: Rc<Owner<T, D>>,
    constructor: Constructor<T, D>,
    bytes: Box<dyn Fn(&D) -> u64>,
}
impl<T, D: BufferDescription> Debug for BufferPool<T, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool").finish()
    }
}
impl<T: 'static, D: BufferDescription> BufferPool<T, D> {
    pub fn new(
        budget: Budget,
        kind: usize,
        bytes: Box<dyn Fn(&D) -> u64>,
        constructor: Constructor<T, D>,
    ) -> Self {
        let key = budget.key();
        let stock = (!budget.limited()).then(Default::default);
        Self {
            owner: Rc::new(Owner {
                budget,
                key,
                kind,
                stock,
            }),
            constructor,
            bytes,
        }
    }
    pub fn take(&self, descriptors: &Descriptors, description: D) -> PoolEntry<T, D> {
        let cached = if let Some(stock) = &self.owner.stock {
            let mut v = stock.borrow_mut();
            let best = v
                .iter()
                .enumerate()
                .filter_map(|(i, (_, d, _))| description.cost_to_use(d).map(|c| (c, i)))
                .min_by(|a, b| a.0.cmp(&b.0));
            best.map(|(_, i)| {
                let (t, d, b) = v.swap_remove(i);
                self.owner.budget.stock_delta(self.owner.kind, 0, b);
                (t, d)
            })
        } else {
            self.owner
                .budget
                .take(self.owner.key, |v: &(T, D)| description.cost_to_use(&v.1))
        };
        let (item, description) =
            cached.unwrap_or_else(|| ((self.constructor)(descriptors, &description), description));
        let bytes = (self.bytes)(&description);
        PoolEntry {
            item: Some(item),
            description,
            owner: Rc::downgrade(&self.owner),
            bytes,
        }
    }
}
pub struct PoolEntry<T: 'static, D: BufferDescription> {
    item: Option<T>,
    description: D,
    owner: Weak<Owner<T, D>>,
    bytes: u64,
}
impl<T: PartialEq, D: BufferDescription> PartialEq for PoolEntry<T, D> {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.owner, &other.owner) && self.item == other.item
    }
}
impl<T: Debug, D: BufferDescription> Debug for PoolEntry<T, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PoolEntry").field(&self.item).finish()
    }
}
impl<T, D: BufferDescription> Deref for PoolEntry<T, D> {
    type Target = T;
    fn deref(&self) -> &T {
        self.item.as_ref().unwrap()
    }
}
impl<T, D: BufferDescription> Drop for PoolEntry<T, D> {
    fn drop(&mut self) {
        if let (Some(item), Some(owner)) = (self.item.take(), self.owner.upgrade()) {
            if let Some(stock) = &owner.stock {
                stock
                    .borrow_mut()
                    .push((item, self.description.clone(), self.bytes));
                owner.budget.stock_delta(owner.kind, self.bytes, 0);
            } else {
                owner.budget.put(
                    owner.key,
                    owner.kind,
                    self.bytes,
                    (item, self.description.clone()),
                );
            }
        }
    }
}
