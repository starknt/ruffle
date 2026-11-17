//! Exact dirty coverage: one rectangle normally, one bit per pixel only when
//! fragmented. Never upload bounding-box holes over authoritative GPU pixels.
use ruffle_render::bitmap::PixelRegion;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Debug)]
pub struct Coverage {
    bounds: PixelRegion,
    mask: Option<Arc<Vec<u64>>>,
    width: u32,
    height: u32,
}
impl Coverage {
    pub fn new(region: PixelRegion, width: u32, height: u32) -> Self {
        Self {
            bounds: region,
            mask: None,
            width,
            height,
        }
    }
    fn mark(mask: &mut [u64], stride: usize, region: PixelRegion) {
        if region.is_empty() {
            return;
        }
        let first = region.x_min / 64;
        let last = (region.x_max - 1) / 64;
        for y in region.y_min..region.y_max {
            for word in first..=last {
                let left = if word == first { region.x_min % 64 } else { 0 };
                let right = if word == last {
                    (region.x_max - 1) % 64 + 1
                } else {
                    64
                };
                mask[y as usize * stride + word as usize] |=
                    (!0u64 << left) & (!0u64 >> (64 - right));
            }
        }
    }
    pub fn add(&mut self, region: PixelRegion) {
        if region.is_empty() {
            return;
        }
        let mut bounds = self.bounds;
        bounds.union(region);
        let area = |r: PixelRegion| u64::from(r.width()) * u64::from(r.height());
        let overlap = u64::from(
            self.bounds
                .x_max
                .min(region.x_max)
                .saturating_sub(self.bounds.x_min.max(region.x_min)),
        ) * u64::from(
            self.bounds
                .y_max
                .min(region.y_max)
                .saturating_sub(self.bounds.y_min.max(region.y_min)),
        );
        if self.mask.is_none() && area(bounds) == area(self.bounds) + area(region) - overlap {
            self.bounds = bounds;
            return;
        }
        let stride = self.width.div_ceil(64) as usize;
        let mask = self.mask.get_or_insert_with(|| {
            let mut mask = vec![0; stride * self.height as usize];
            Self::mark(&mut mask, stride, self.bounds);
            Arc::new(mask)
        });
        Self::mark(Arc::make_mut(mask).as_mut_slice(), stride, region);
        self.bounds = bounds;
    }
    pub fn for_each<E>(
        &self,
        mut visit: impl FnMut(PixelRegion) -> Result<(), E>,
    ) -> Result<(), E> {
        let Some(mask) = &self.mask else {
            return visit(self.bounds);
        };
        let stride = self.width.div_ceil(64) as usize;
        let mut active = BTreeMap::<(u32, u32), PixelRegion>::new();
        for y in self.bounds.y_min..self.bounds.y_max {
            let dirty =
                |x: u32| mask[y as usize * stride + (x / 64) as usize] & (1u64 << (x % 64)) != 0;
            let mut current = BTreeMap::new();
            let mut x = self.bounds.x_min;
            while x < self.bounds.x_max {
                if !dirty(x) {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < self.bounds.x_max && dirty(x) {
                    x += 1;
                }
                let mut region = active
                    .remove(&(start, x))
                    .unwrap_or(PixelRegion::for_region(start, y, x - start, 1));
                region.y_max = y + 1;
                current.insert((start, x), region);
            }
            for region in active.into_values() {
                visit(region)?;
            }
            active = current;
        }
        for region in active.into_values() {
            visit(region)?;
        }
        Ok(())
    }
}
