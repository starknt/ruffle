use crate::filters::FilterSource;
use ruffle_render::bitmap::PixelRegion;
use ruffle_render::filters::Filter;
use swf::{Rectangle, Twips};

/// Return the source-space origin and destination pixels of the clipped gradient filter.
pub(super) fn output_region(
    source: FilterSource<'_>,
    dest_point: (i32, i32),
    dest_size: (u32, u32),
    filter: &Filter,
) -> Option<([i32; 2], PixelRegion)> {
    let mut src =
        PixelRegion::for_region(source.point.0, source.point.1, source.size.0, source.size.1);
    src.clamp(source.texture.width(), source.texture.height());
    if src.is_empty() {
        return None;
    }
    let bounds = filter.calculate_dest_rect(
        Rectangle::ZERO
            .with_width(Twips::from_pixels(f64::from(src.width())))
            .with_height(Twips::from_pixels(f64::from(src.height()))),
    );
    let (dx, dy) = (f64::from(dest_point.0), f64::from(dest_point.1));
    let mut destination = PixelRegion {
        x_min: (dx + bounds.x_min.to_pixels()).floor() as u32,
        y_min: (dy + bounds.y_min.to_pixels()).floor() as u32,
        x_max: (dx + bounds.x_max.to_pixels()).ceil() as u32,
        y_max: (dy + bounds.y_max.to_pixels()).ceil() as u32,
    };
    destination.clamp(dest_size.0, dest_size.1);
    if destination.is_empty() {
        return None;
    }
    let origin = [
        src.x_min as i32 + (destination.x_min as i32 - dest_point.0),
        src.y_min as i32 + (destination.y_min as i32 - dest_point.1),
    ];
    Some((origin, destination))
}
