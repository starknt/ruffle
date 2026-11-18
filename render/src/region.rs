//! Conservative device-space coverage. Unknown dependencies dominate finite bounds.
use crate::matrix::Matrix;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum RenderRegion {
    #[default]
    Empty,
    Finite([f64; 4]),
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl RenderRegion {
    pub fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::Full, _) | (_, Self::Full) => Self::Full,
            (Self::Empty, r) | (r, Self::Empty) => r,
            (Self::Finite(a), Self::Finite(b)) => Self::Finite([
                a[0].min(b[0]),
                a[1].min(b[1]),
                a[2].max(b[2]),
                a[3].max(b[3]),
            ]),
        }
    }

    /// Geometry includes tessellated strokes. Pad in device pixels for AA and
    /// rasterization rounding, after applying the complete affine transform.
    pub fn transformed(bounds: [f64; 4], matrix: &Matrix) -> Self {
        let mut result = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for x in [bounds[0], bounds[2]] {
            for y in [bounds[1], bounds[3]] {
                let px = x * f64::from(matrix.a) + y * f64::from(matrix.c) + matrix.tx.to_pixels();
                let py = x * f64::from(matrix.b) + y * f64::from(matrix.d) + matrix.ty.to_pixels();
                if !px.is_finite() || !py.is_finite() {
                    return Self::Full;
                }
                result[0] = result[0].min(px);
                result[1] = result[1].min(py);
                result[2] = result[2].max(px);
                result[3] = result[3].max(py);
            }
        }
        Self::Finite([
            result[0] - 2.0,
            result[1] - 2.0,
            result[2] + 2.0,
            result[3] + 2.0,
        ])
    }

    pub fn clip(self, viewport: RenderRect) -> Option<RenderRect> {
        let Self::Finite(b) = self else {
            return (self == Self::Full).then_some(viewport);
        };
        if b.iter().any(|x| !x.is_finite()) {
            return Some(viewport);
        }
        let x = (b[0].floor().max(f64::from(viewport.x)) as u32).min(viewport.x + viewport.width);
        let y = (b[1].floor().max(f64::from(viewport.y)) as u32).min(viewport.y + viewport.height);
        let right =
            (b[2].ceil().max(f64::from(viewport.x)) as u32).min(viewport.x + viewport.width);
        let bottom =
            (b[3].ceil().max(f64::from(viewport.y)) as u32).min(viewport.y + viewport.height);
        (right > x && bottom > y).then_some(RenderRect {
            x,
            y,
            width: right.saturating_sub(x),
            height: bottom.saturating_sub(y),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coverage_rounds_out_and_clips_negative_coordinates() {
        let viewport = RenderRect {
            x: 0,
            y: 0,
            width: 64,
            height: 64,
        };
        assert_eq!(
            RenderRegion::Finite([-4.5, 1.5, 8.2, 9.1]).clip(viewport),
            Some(RenderRect {
                x: 0,
                y: 1,
                width: 9,
                height: 9
            })
        );
        assert_eq!(RenderRegion::Empty.clip(viewport), None);
        assert_eq!(
            RenderRegion::Full.union(RenderRegion::Empty).clip(viewport),
            Some(viewport)
        );
        assert_eq!(
            RenderRegion::Finite([-9.0, -9.0, -1.0, -1.0]).clip(viewport),
            None
        );
    }
}
