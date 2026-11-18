//! Exact dirty coverage: one rectangle normally, one bit per pixel only when
//! fragmented. Never upload bounding-box holes over authoritative GPU pixels.
use super::PixelRegion;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Debug)]
pub struct Coverage {
    bounds: PixelRegion,
    pixels: u64,
    mask: Option<Arc<Vec<u64>>>,
    width: u32,
    height: u32,
}
impl Coverage {
    pub fn new(region: PixelRegion, width: u32, height: u32) -> Self {
        Self {
            bounds: region,
            pixels: u64::from(region.width()) * u64::from(region.height()),
            mask: None,
            width,
            height,
        }
    }
    pub fn empty(width: u32, height: u32) -> Self {
        Self::new(PixelRegion::for_whole_size(0, 0), width, height)
    }
    pub fn bounds(&self) -> PixelRegion {
        self.bounds
    }
    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }
    fn mark(mask: &mut [u64], stride: usize, region: PixelRegion, value: bool) -> u64 {
        if region.is_empty() {
            return 0;
        }
        let mut changed = 0;
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
                let bits = (!0u64 << left) & (!0u64 >> (64 - right));
                let word = &mut mask[y as usize * stride + word as usize];
                changed +=
                    u64::from((if value { !*word & bits } else { *word & bits }).count_ones());
                if value {
                    *word |= bits;
                } else {
                    *word &= !bits;
                }
            }
        }
        changed
    }
    pub fn add(&mut self, region: PixelRegion) {
        if region.is_empty() {
            return;
        }
        if self.is_empty() {
            self.bounds = region;
            self.pixels = u64::from(region.width()) * u64::from(region.height());
            self.mask = None;
            return;
        }
        debug_assert!(region.x_max <= self.width && region.y_max <= self.height);
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
            self.pixels = area(bounds);
            return;
        }
        let stride = self.width.div_ceil(64) as usize;
        let mask = self.mask.get_or_insert_with(|| {
            let mut mask = vec![0; stride * self.height as usize];
            Self::mark(&mut mask, stride, self.bounds, true);
            Arc::new(mask)
        });
        self.pixels += Self::mark(Arc::make_mut(mask).as_mut_slice(), stride, region, true);
        self.bounds = bounds;
    }
    pub fn remove(&mut self, region: PixelRegion) {
        let region = intersection(region, self.bounds);
        if region.is_empty() {
            return;
        }
        if region == self.bounds {
            *self = Self::empty(self.width, self.height);
            return;
        }
        let stride = self.width.div_ceil(64) as usize;
        let mask = self.mask.get_or_insert_with(|| {
            let mut mask = vec![0; stride * self.height as usize];
            Self::mark(&mut mask, stride, self.bounds, true);
            Arc::new(mask)
        });
        self.pixels -= Self::mark(Arc::make_mut(mask).as_mut_slice(), stride, region, false);
        if self.pixels == 0 {
            *self = Self::empty(self.width, self.height);
        }
    }
    pub fn intersects(&self, region: PixelRegion) -> bool {
        let region = intersection(region, self.bounds);
        if region.is_empty() {
            return false;
        }
        let Some(mask) = &self.mask else {
            return true;
        };
        let stride = self.width.div_ceil(64) as usize;
        for y in region.y_min..region.y_max {
            for word in region.x_min / 64..region.x_max.div_ceil(64) {
                let left = region.x_min.saturating_sub(word * 64);
                let right = (region.x_max - word * 64).min(64);
                let bits = (!0u64 << left) & (!0u64 >> (64 - right));
                if mask[y as usize * stride + word as usize] & bits != 0 {
                    return true;
                }
            }
        }
        false
    }
    pub fn for_each<E>(&self, visit: impl FnMut(PixelRegion) -> Result<(), E>) -> Result<(), E> {
        self.for_each_in(self.bounds, visit)
    }
    pub fn for_each_in<E>(
        &self,
        region: PixelRegion,
        mut visit: impl FnMut(PixelRegion) -> Result<(), E>,
    ) -> Result<(), E> {
        let bounds = intersection(self.bounds, region);
        if bounds.is_empty() {
            return Ok(());
        }
        let Some(mask) = &self.mask else {
            return visit(bounds);
        };
        let stride = self.width.div_ceil(64) as usize;
        let mut active = BTreeMap::<(u32, u32), PixelRegion>::new();
        for y in bounds.y_min..bounds.y_max {
            let dirty =
                |x: u32| mask[y as usize * stride + (x / 64) as usize] & (1u64 << (x % 64)) != 0;
            let mut current = BTreeMap::new();
            let mut x = bounds.x_min;
            while x < bounds.x_max {
                if !dirty(x) {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < bounds.x_max && dirty(x) {
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

fn intersection(a: PixelRegion, b: PixelRegion) -> PixelRegion {
    let x = a.x_min.max(b.x_min);
    let y = a.y_min.max(b.y_min);
    PixelRegion::for_region(
        x,
        y,
        a.x_max.min(b.x_max).saturating_sub(x),
        a.y_max.min(b.y_max).saturating_sub(y),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragmented_coverage_never_merges_unwritten_pixels_across_words_or_rows() {
        let mut coverage = Coverage::empty(130, 3);
        let mut expected = vec![false; 390];
        let operations = [
            (true, (62, 0, 6, 3)),
            (false, (63, 1, 2, 1)),
            (true, (129, 2, 1, 1)),
            (true, (0, 0, 1, 1)),
            (false, (0, 0, 130, 1)),
            (true, (64, 1, 1, 1)),
            (false, (62, 0, 6, 3)),
            (false, (129, 2, 1, 1)),
        ];
        for (add, (x, y, width, height)) in operations {
            let region = PixelRegion::for_region(x, y, width, height);
            if add {
                coverage.add(region);
            } else {
                coverage.remove(region);
            }
            for row in y..y + height {
                for col in x..x + width {
                    expected[(row * 130 + col) as usize] = add;
                }
            }
            let mut actual = vec![false; 390];
            coverage
                .for_each::<()>(|rect| {
                    for row in rect.y_min..rect.y_max {
                        for col in rect.x_min..rect.x_max {
                            assert!(
                                !actual[(row * 130 + col) as usize],
                                "coverage rectangles overlap"
                            );
                            actual[(row * 130 + col) as usize] = true;
                        }
                    }
                    Ok(())
                })
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(coverage.is_empty(), !expected.iter().any(|&set| set));
            for row in 0..3 {
                for col in 0..130 {
                    assert_eq!(
                        coverage.intersects(PixelRegion::for_pixel(col, row)),
                        expected[(row * 130 + col) as usize]
                    );
                }
            }
        }
    }
}
