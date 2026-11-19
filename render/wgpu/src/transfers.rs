//! BitmapData barriers lower explicit numeric copy/readback graphs on the owner
//! thread. Native queue order connects these graphs to preceding/following AS ops.
use crate::{
    as_texture,
    buffer_pool::TexturePool,
    descriptors::Descriptors,
    gpu_resources::{Buffer, CommandEncoder},
    gpu_runtime::{RenderFailure, RenderFailureKind},
    pass_graph::{PassGraph, PassKind},
    render_snapshot::bitmap_version,
    utils::BufferDimensions,
};
use ruffle_render::{
    bitmap::{BitmapHandle, PixelRegion},
    region::RenderRect,
};

fn rect(region: PixelRegion) -> RenderRect {
    RenderRect {
        x: region.x_min,
        y: region.y_min,
        width: region.width(),
        height: region.height(),
    }
}
fn origin(rect: RenderRect) -> wgpu::Origin3d {
    wgpu::Origin3d {
        x: rect.x,
        y: rect.y,
        z: 0,
    }
}

pub fn copy(
    descriptors: &Descriptors,
    pool: &mut TexturePool,
    encoder: &mut CommandEncoder,
    source: &BitmapHandle,
    source_region: PixelRegion,
    destination: &BitmapHandle,
    destination_region: PixelRegion,
) -> Result<(), RenderFailure> {
    let source_texture = as_texture(source, &descriptors.device)?;
    let destination_texture = as_texture(destination, &descriptors.device)?;
    let mut graph = PassGraph::default();
    let source_key = graph.import(bitmap_version(source)?);
    let destination_key = graph.import(bitmap_version(destination)?);
    let source_rect = rect(source_region);
    let destination_rect = rect(destination_region);
    let temporary;
    let temporary_key;
    if source_texture.texture == destination_texture.texture {
        let bounds = RenderRect {
            x: 0,
            y: 0,
            width: source_rect.width,
            height: source_rect.height,
        };
        temporary_key = Some(graph.temporary(bounds, 1)?);
        graph.copy_region(source_key, temporary_key.unwrap(), source_rect, bounds)?;
        graph.copy_region(
            temporary_key.unwrap(),
            destination_key,
            bounds,
            destination_rect,
        )?;
        // Admission precedes both writes; self-copy has memmove semantics.
        temporary = Some(pool.get_texture(
            descriptors,
            wgpu::Extent3d {
                width: bounds.width,
                height: bounds.height,
                depth_or_array_layers: 1,
            },
            wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            source_texture.texture.format(),
            1,
        )?);
    } else {
        temporary = None;
        temporary_key = None;
        graph.copy_region(source_key, destination_key, source_rect, destination_rect)?;
    }
    for (index, pass) in graph.passes.iter().enumerate() {
        if pass
            .dependencies
            .iter()
            .any(|&dependency| dependency >= index)
        {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "invalid copy graph dependency",
            ));
        }
        let PassKind::Copy { source_bounds } = pass.kind else {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "non-copy node in copy graph",
            ));
        };
        let input = if Some(pass.reads[0].resource) == temporary_key {
            &temporary.as_ref().unwrap().0
        } else {
            &source_texture.texture
        };
        let output = if Some(pass.writes[0].resource) == temporary_key {
            &temporary.as_ref().unwrap().0
        } else {
            &destination_texture.texture
        };
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: input,
                mip_level: 0,
                origin: origin(source_bounds),
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: output,
                mip_level: 0,
                origin: origin(pass.bounds),
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: pass.bounds.width,
                height: pass.bounds.height,
                depth_or_array_layers: 1,
            },
        );
    }
    Ok(())
}

pub fn readback(
    descriptors: &Descriptors,
    encoder: &mut CommandEncoder,
    source: &BitmapHandle,
    region: PixelRegion,
    buffer: &Buffer,
    dimensions: &BufferDimensions,
) -> Result<(), RenderFailure> {
    let texture = as_texture(source, &descriptors.device)?;
    let mut graph = PassGraph::default();
    let source_key = graph.import(bitmap_version(source)?);
    graph.readback(source_key, rect(region))?;
    for pass in graph.passes {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.texture,
                mip_level: 0,
                origin: origin(pass.bounds),
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dimensions.padded_bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: pass.bounds.width,
                height: pass.bounds.height,
                depth_or_array_layers: 1,
            },
        );
    }
    Ok(())
}
