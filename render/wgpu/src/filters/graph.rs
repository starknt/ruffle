//! Pure filter preparation: adjacent pointwise groups plus conservative backward
//! sampling footprints. No shader objects or renderer handles cross this boundary.
use ruffle_render::filters::Filter;
use std::ops::Range;
pub const MAX_POINTWISE_FILTERS: usize = 16;
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FilterStep {
    Pointwise(Range<usize>),
    Single(usize),
}

impl FilterStep {
    pub fn profile_label(&self, filters: &[Filter]) -> &'static str {
        let Self::Single(index) = self else {
            return "ColorMatrixFilter";
        };
        filter_label(&filters[*index])
    }
}
pub(crate) fn filter_label(filter: &Filter) -> &'static str {
    match filter {
        Filter::BlurFilter(_) => "BlurFilter",
        Filter::ColorMatrixFilter(_) => "ColorMatrixFilter",
        Filter::GlowFilter(_) => "GlowFilter",
        Filter::BevelFilter(_) => "BevelFilter",
        Filter::DropShadowFilter(_) => "DropShadowFilter",
        Filter::DisplacementMapFilter(_) => "DisplacementMapFilter",
        Filter::ConvolutionFilter(_) => "ConvolutionFilter",
        Filter::GradientGlowFilter(_) => "GradientGlowFilter",
        Filter::GradientBevelFilter(_) => "GradientBevelFilter",
        Filter::ShaderFilter(_) => "ShaderFilter",
    }
}
#[derive(Clone, Copy, Debug)]
enum Sampling {
    Local([i64; 4]),
    Full,
}
#[derive(Clone, Debug)]
pub struct FilterGraph {
    pub steps: Vec<FilterStep>,
    sampling: Vec<Sampling>,
}
impl FilterGraph {
    pub fn new(filters: &[Filter], pointwise: bool) -> Self {
        let eligible = |filter: &Filter| matches!(filter, Filter::ColorMatrixFilter(matrix) if pointwise && matrix.matrix.iter().all(|value| value.is_finite()));
        let mut steps = Vec::new();
        let mut index = 0;
        while index < filters.len() {
            if eligible(&filters[index]) {
                let start = index;
                index += 1;
                // Only byte-preserving channel permutations may lose their
                // attachment. General matrices retain the backend's RGBA8
                // conversion, which WGSL pack/round cannot portably reproduce.
                while index < filters.len()
                    && index - start < MAX_POINTWISE_FILTERS
                    && eligible(&filters[index])
                {
                    let Filter::ColorMatrixFilter(previous) = &filters[index - 1] else {
                        break;
                    };
                    if channel_permutation(previous).is_none() {
                        break;
                    }
                    index += 1;
                }
                steps.push(FilterStep::Pointwise(start..index));
            } else {
                steps.push(FilterStep::Single(index));
                index += 1;
            }
        }
        Self {
            steps,
            sampling: filters.iter().map(sampling).collect(),
        }
    }
    pub fn required_input(&self, mut output: [i64; 4], valid: [i64; 4]) -> [i64; 4] {
        for sample in self.sampling.iter().rev() {
            let Sampling::Local(halo) = sample else {
                return valid;
            };
            output = [
                output[0].saturating_sub(halo[0]),
                output[1].saturating_sub(halo[1]),
                output[2].saturating_add(halo[2]),
                output[3].saturating_add(halo[3]),
            ];
        }
        clip(output, valid)
    }
    pub fn affected_output(&self, mut input: [i64; 4], valid: [i64; 4]) -> [i64; 4] {
        for sample in &self.sampling {
            let Sampling::Local(halo) = sample else {
                return valid;
            };
            input = [
                input[0].saturating_sub(halo[2]),
                input[1].saturating_sub(halo[3]),
                input[2].saturating_add(halo[0]),
                input[3].saturating_add(halo[1]),
            ];
        }
        clip(input, valid)
    }
}
/// Each intermediate maps an RGBA8 channel back to the same byte. Carry the
/// sampled channel directly, avoiding a second, backend-specific UNORM rounding.
pub fn channel_permutation(filter: &swf::ColorMatrixFilter) -> Option<[u32; 3]> {
    if filter.matrix[15..] != [0.0, 0.0, 0.0, 1.0, 0.0] {
        return None;
    }
    let mut channels = [0; 3];
    for (index, row) in filter.matrix[..15].as_chunks::<5>().0.iter().enumerate() {
        let channel = row[..3].iter().position(|&value| value == 1.0)?;
        if row
            .iter()
            .enumerate()
            .any(|(i, &value)| value != if i == channel { 1.0 } else { 0.0 })
        {
            return None;
        }
        channels[index] = channel as u32;
    }
    Some(channels)
}

fn clip(region: [i64; 4], valid: [i64; 4]) -> [i64; 4] {
    [
        region[0].clamp(valid[0], valid[2]),
        region[1].clamp(valid[1], valid[3]),
        region[2].clamp(valid[0], valid[2]),
        region[3].clamp(valid[1], valid[3]),
    ]
}
fn blur(filter: &swf::BlurFilter) -> [i64; 4] {
    let radius = |size: swf::Fixed16| {
        (((size.to_f32().min(255.0) - 1.0) * 0.5).max(0.0).ceil() as i64)
            * i64::from(filter.num_passes())
    };
    let x = radius(filter.blur_x);
    let y = radius(filter.blur_y);
    [x, y, x, y]
}
fn sampling(filter: &Filter) -> Sampling {
    Sampling::Local(match filter {
        Filter::ColorMatrixFilter(_) => [0; 4],
        Filter::BlurFilter(filter) => blur(filter),
        Filter::GlowFilter(filter) => blur(&filter.inner_blur_filter()),
        Filter::DropShadowFilter(filter) => {
            let mut halo = blur(&filter.inner_blur_filter());
            let x = filter.angle.to_f32().cos() * filter.distance.to_f32();
            let y = filter.angle.to_f32().sin() * filter.distance.to_f32();
            for (h, amount) in
                halo.iter_mut()
                    .zip([x.max(0.0), y.max(0.0), (-x).max(0.0), (-y).max(0.0)])
            {
                *h += amount.ceil() as i64;
            }
            halo
        }
        Filter::BevelFilter(filter) => {
            let mut halo = blur(&filter.inner_blur_filter());
            let x = (filter.angle.to_f32().cos() * filter.distance.to_f32())
                .abs()
                .ceil() as i64;
            let y = (filter.angle.to_f32().sin() * filter.distance.to_f32())
                .abs()
                .ceil() as i64;
            for (h, amount) in halo.iter_mut().zip([x, y, x, y]) {
                *h += amount;
            }
            halo
        }
        // Displacement may wrap to the opposite edge; arbitrary shaders can
        // read any coordinate. Keep the complete valid source in both cases.
        _ => return Sampling::Full,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruffle_render::filters::Filter;
    #[test]
    fn filter_graph_backward_sampling_accumulates_blur_halo_and_clips_once() {
        let filters = vec![
            Filter::BlurFilter(swf::BlurFilter {
                blur_x: swf::Fixed16::from_f32(5.0),
                blur_y: swf::Fixed16::from_f32(3.0),
                flags: swf::BlurFilterFlags::from_passes(2),
            }),
            Filter::ColorMatrixFilter(Default::default()),
            Filter::BlurFilter(swf::BlurFilter {
                blur_x: swf::Fixed16::from_f32(2.5),
                blur_y: swf::Fixed16::ONE,
                flags: swf::BlurFilterFlags::from_passes(1),
            }),
        ];
        let graph = FilterGraph::new(&filters, true);
        assert_eq!(
            graph.required_input([10, 12, 20, 24], [0, 0, 64, 64]),
            [5, 10, 25, 26]
        );
        assert_eq!(
            graph.required_input([1, 1, 3, 3], [0, 0, 64, 64]),
            [0, 0, 8, 5]
        );
    }
    #[test]
    fn filter_graph_does_not_fuse_across_sampling_operations_or_nonfinite_matrices() {
        let mut invalid = swf::ColorMatrixFilter::default();
        invalid.matrix[0] = f32::NAN;
        let filters = vec![
            Filter::default(),
            Filter::default(),
            Filter::BlurFilter(swf::BlurFilter {
                blur_x: swf::Fixed16::from_f32(5.0),
                blur_y: swf::Fixed16::ONE,
                flags: swf::BlurFilterFlags::from_passes(1),
            }),
            Filter::ColorMatrixFilter(invalid),
            Filter::default(),
        ];
        let graph = FilterGraph::new(&filters, true);
        assert_eq!(
            graph.steps,
            vec![
                FilterStep::Pointwise(0..2),
                FilterStep::Single(2),
                FilterStep::Single(3),
                FilterStep::Pointwise(4..5)
            ]
        );
    }
    #[test]
    fn filter_graph_keeps_general_matrix_quantization_boundaries() {
        let mut matrix = swf::ColorMatrixFilter::default();
        matrix.matrix[0] = 1.13;
        matrix.matrix[4] = 0.5;
        let graph = FilterGraph::new(
            &[
                Filter::ColorMatrixFilter(matrix.clone()),
                Filter::ColorMatrixFilter(matrix),
            ],
            true,
        );
        assert_eq!(
            graph.steps,
            vec![FilterStep::Pointwise(0..1), FilterStep::Pointwise(1..2)]
        );
    }
}
