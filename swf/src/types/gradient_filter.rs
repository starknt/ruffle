use crate::{BlurFilter, BlurFilterFlags, Fixed8, Fixed16, GradientRecord, Rectangle, Twips};
use bitflags::bitflags;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GradientFilter {
    pub colors: Vec<GradientRecord>,
    pub blur_x: Fixed16,
    pub blur_y: Fixed16,
    pub angle: Fixed16,
    pub distance: Fixed16,
    pub strength: Fixed8,
    pub flags: GradientFilterFlags,
}

impl GradientFilter {
    #[inline]
    pub fn is_inner(&self) -> bool {
        self.flags.contains(GradientFilterFlags::INNER_SHADOW)
    }

    #[inline]
    pub fn is_knockout(&self) -> bool {
        self.flags.contains(GradientFilterFlags::KNOCKOUT)
    }

    #[inline]
    pub fn is_on_top(&self) -> bool {
        self.flags.contains(GradientFilterFlags::ON_TOP)
    }

    #[inline]
    pub fn num_passes(&self) -> u8 {
        (self.flags & GradientFilterFlags::PASSES).bits()
    }

    pub fn scale(&mut self, x: f32, y: f32) {
        self.blur_x = BlurFilter::scale_blur(self.blur_x, x);
        self.blur_y = BlurFilter::scale_blur(self.blur_y, y);
        self.distance *= Fixed16::from_f32(y);
    }

    pub fn calculate_glow_dest_rect(&self, source: Rectangle<Twips>) -> Rectangle<Twips> {
        self.gradient_dest_rect(source, false)
    }

    pub fn calculate_bevel_dest_rect(&self, source: Rectangle<Twips>) -> Rectangle<Twips> {
        self.gradient_dest_rect(source, true)
    }

    fn gradient_dest_rect(&self, source: Rectangle<Twips>, bevel: bool) -> Rectangle<Twips> {
        if !source.is_valid()
            || self.is_inner()
            || self.colors.is_empty()
            || (!bevel && self.strength <= Fixed8::ZERO)
        {
            return source;
        }
        let blur = self.inner_blur_filter();
        let halo = if blur.impotent() {
            Rectangle::ZERO
        } else {
            blur.calculate_dest_rect(Rectangle::ZERO)
        };
        let (sin, cos) = self.angle.to_f32().sin_cos();
        let distance = self.distance.to_f32();
        // Sample positions are fractional. Round outward so even a sub-twip shift can reach the next pixel.
        let dx = f64::from(cos * distance) * f64::from(Twips::TWIPS_PER_PIXEL);
        let dy = f64::from(sin * distance) * f64::from(Twips::TWIPS_PER_PIXEL);
        let shifted = |edge: Twips, radius: Twips, offset: f64| {
            Twips::new(edge.get().saturating_add(radius.get() + offset as i32))
        };
        // A bevel takes samples in both directions along the light axis.
        let offsets = |d: f64| if bevel { (-d.abs(), d.abs()) } else { (d, d) };
        let (left, right) = offsets(dx);
        let (top, bottom) = offsets(dy);
        source.union(&Rectangle {
            x_min: shifted(source.x_min, halo.x_min, left.floor()),
            x_max: shifted(source.x_max, halo.x_max, right.ceil()),
            y_min: shifted(source.y_min, halo.y_min, top.floor()),
            y_max: shifted(source.y_max, halo.y_max, bottom.ceil()),
        })
    }

    pub fn inner_blur_filter(&self) -> BlurFilter {
        BlurFilter {
            blur_x: self.blur_x,
            blur_y: self.blur_y,
            flags: BlurFilterFlags::from_passes(self.num_passes()),
        }
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct GradientFilterFlags: u8 {
        const INNER_SHADOW     = 1 << 7;
        const KNOCKOUT         = 1 << 6;
        const COMPOSITE_SOURCE = 1 << 5;
        const ON_TOP           = 1 << 4;
        const PASSES           = 0b1111;
    }
}

impl GradientFilterFlags {
    #[inline]
    pub fn from_passes(num_passes: u8) -> Self {
        let flags = Self::from_bits_retain(num_passes);
        debug_assert_eq!(flags & Self::PASSES, flags);
        flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Color;

    #[test]
    fn glow_bounds_cover_the_shifted_blur_and_original_image() {
        let px = Twips::from_pixels_i32;
        let source = Rectangle {
            x_min: px(10),
            x_max: px(20),
            y_min: px(20),
            y_max: px(30),
        };
        let mut filter = GradientFilter {
            colors: vec![GradientRecord {
                ratio: 255,
                color: Color::WHITE,
            }],
            blur_x: Fixed16::from_f32(3.0),
            blur_y: Fixed16::from_f32(2.0),
            angle: Fixed16::ZERO,
            distance: Fixed16::from_f32(-6.0),
            strength: Fixed8::ONE,
            flags: GradientFilterFlags::from_passes(1) | GradientFilterFlags::ON_TOP,
        };
        assert_eq!(
            filter.calculate_glow_dest_rect(source),
            Rectangle {
                x_min: px(1),
                x_max: px(20),
                y_min: px(18),
                y_max: px(32),
            }
        );
        filter.flags.insert(GradientFilterFlags::INNER_SHADOW);
        assert_eq!(filter.calculate_glow_dest_rect(source), source);

        filter.flags.remove(GradientFilterFlags::INNER_SHADOW);
        filter.blur_x = Fixed16::ONE;
        filter.blur_y = Fixed16::ONE;
        for (distance, left, right) in [(1.0 / 32.0, 0, 1), (-1.0 / 32.0, -1, 0)] {
            filter.distance = Fixed16::from_f32(distance);
            let bounds = filter.calculate_glow_dest_rect(source);
            assert_eq!(bounds.x_min.get(), source.x_min.get() + left);
            assert_eq!(bounds.x_max.get(), source.x_max.get() + right);
            assert_eq!((bounds.y_min, bounds.y_max), (source.y_min, source.y_max));
            let bevel = filter.calculate_bevel_dest_rect(source);
            assert_eq!(bevel.x_min.get(), source.x_min.get() - 1);
            assert_eq!(bevel.x_max.get(), source.x_max.get() + 1);
        }
    }
}
