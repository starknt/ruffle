use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::filters::FilterSource;
use crate::filters::blur::BlurFilter;
use crate::filters::glow::GlowFilter;
use crate::gpu_resources::StagingBelt;
use crate::surface::target::CommandTarget;
use swf::DropShadowFilter as DropShadowFilterArgs;

/// Drop shadow is just Glow with an offset.
/// None of this strictly needs to be a struct,
/// but it helps for code organisation + if we want to specialise the implementation in the future
pub struct DropShadowFilter;

impl DropShadowFilter {
    #[expect(clippy::too_many_arguments)]
    pub fn apply(
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        draw_encoder: &mut crate::gpu_resources::CommandEncoder,
        staging_belt: &mut StagingBelt,
        source: &FilterSource,
        filter: &DropShadowFilterArgs,
        blur_filter: &BlurFilter,
        glow_filter: &GlowFilter,
    ) -> Result<CommandTarget, crate::gpu_runtime::RenderFailure> {
        let distance = filter.distance.to_f32();
        let angle = filter.angle.to_f32();
        let x = angle.cos() * distance;
        let y = angle.sin() * distance;
        Ok(glow_filter.apply(
            descriptors,
            texture_pool,
            draw_encoder,
            staging_belt,
            source,
            &filter.inner_glow_filter(),
            blur_filter,
            (-x, -y),
        )?)
    }
}
