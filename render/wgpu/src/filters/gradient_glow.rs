use super::{FilterPipelines, FilterSource, GradientKind, VERTEX_BUFFERS_DESCRIPTION_FILTERS};
use crate::backend::RenderTargetMode;
use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::gpu_resources::{
    BindGroupLayout, Buffer, CommandEncoder, Device, PipelineLayout, StagingBelt,
};
use crate::gpu_runtime::{RenderFailure, RenderFailureKind};
use crate::surface::target::CommandTarget;
use std::ops::Range;
use swf::{BlurFilter, Fixed16, GradientFilter, GradientFilterFlags, GradientRecord};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GradientUniform {
    source_point: [i32; 2],
    mask_point: [f32; 2],
    opposite_point: [f32; 2],
    strength: f32,
    mode: u32,
    opaque_target: u32,
    source_mode: u32,
    kind: u32,
    _padding: u32,
    colors: [[u32; 4]; 64],
}

const UNIFORM_SIZE: wgpu::BufferSize =
    wgpu::BufferSize::new(size_of::<GradientUniform>() as u64).unwrap();

fn gradient_colors(stops: &[GradientRecord], kind: GradientKind) -> [[u32; 4]; 64] {
    let mut colors = [[0; 4]; 64];
    let Some((mut left, remaining)) = stops.split_first() else {
        return colors;
    };
    let mut remaining = remaining.iter().peekable();
    for (ratio, color) in (0u8..=255).zip(colors.iter_mut().flatten()) {
        while remaining.peek().is_some_and(|stop| stop.ratio < ratio) {
            left = remaining.next().unwrap();
        }
        let right = remaining.peek().copied().unwrap_or(left);
        let width = u32::from(right.ratio.saturating_sub(left.ratio)).max(1);
        let t = u32::from(ratio.saturating_sub(left.ratio)).min(width);
        let bias = match kind {
            GradientKind::Glow => width / 2,
            GradientKind::Bevel => 0,
        };
        let channel = |a, b| ((u32::from(a) * (width - t) + u32::from(b) * t + bias) / width) as u8;
        let (a, b) = (left.color, right.color);
        *color = u32::from_le_bytes([
            channel(a.r, b.r),
            channel(a.g, b.g),
            channel(a.b, b.b),
            channel(a.a, b.a),
        ]);
    }
    colors
}

// Include both bilinear lookup texels and the support of every blur pass. Beyond
// the bitmap plus that halo, every sample is transparent regardless of offset.
fn mask_axis(
    origin: i32,
    size: u32,
    offset: f32,
    blur: Fixed16,
    passes: u8,
    bitmap_size: u32,
) -> Option<Range<i32>> {
    let radius = ((blur.to_f32().clamp(1.0, 255.0) - 1.0) / 2.0).ceil() as i32 * i32::from(passes);
    let start = origin - offset.ceil() as i32 - radius;
    let end = origin + size as i32 - offset.floor() as i32 + radius;
    let bitmap_end = bitmap_size as i32;
    (start < bitmap_end && end > 0).then(|| start.max(-radius)..end.min(bitmap_end + radius))
}

// Share a blur when its bounding rectangle costs at most the two cropped blurs.
fn shared_mask_region(
    a: &[Range<i32>; 2],
    b: &[Range<i32>; 2],
    limit: u32,
) -> Option<[Range<i32>; 2]> {
    let union: [Range<i32>; 2] =
        std::array::from_fn(|i| a[i].start.min(b[i].start)..a[i].end.max(b[i].end));
    let area = |r: &[Range<i32>; 2]| r[0].len() as u64 * r[1].len() as u64;
    (union.iter().all(|axis| axis.len() <= limit as usize) && area(&union) <= area(a) + area(b))
        .then_some(union)
}

fn blurred_mask(
    descriptors: &Descriptors,
    texture_pool: &mut TexturePool,
    encoder: &mut CommandEncoder,
    staging_belt: &mut StagingBelt,
    source: &FilterSource,
    region: Option<[Range<i32>; 2]>,
    blur: &BlurFilter,
) -> Result<Option<(CommandTarget, [i32; 2])>, RenderFailure> {
    let Some([x, y]) = region else {
        return Ok(None);
    };
    let width = source.texture.width();
    let height = source.texture.height();
    let extent = wgpu::Extent3d {
        width: (x.end - x.start) as u32,
        height: (y.end - y.start) as u32,
        depth_or_array_layers: 1,
    };
    let limit = descriptors.limits.max_texture_dimension_2d;
    if extent.width > limit || extent.height > limit {
        return Err(RenderFailure::new(
            RenderFailureKind::Validation,
            "gradient filter blur mask exceeds device texture limits",
        ));
    }
    let padded = CommandTarget::new(
        descriptors,
        texture_pool,
        extent,
        source.texture.format(),
        1,
        RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
        encoder,
    )?;
    padded.ensure_cleared(encoder);
    let left = x.start.max(0);
    let top = y.start.max(0);
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            origin: wgpu::Origin3d {
                x: left as u32,
                y: top as u32,
                z: 0,
            },
            ..source.texture.as_image_copy()
        },
        wgpu::TexelCopyTextureInfo {
            origin: wgpu::Origin3d {
                x: (left - x.start) as u32,
                y: (top - y.start) as u32,
                z: 0,
            },
            ..padded.color_texture().as_image_copy()
        },
        wgpu::Extent3d {
            width: (x.end.min(width as i32) - left) as u32,
            height: (y.end.min(height as i32) - top) as u32,
            depth_or_array_layers: 1,
        },
    );
    let input = FilterSource::for_entire_texture(padded.color_texture(), padded.color_view());
    let mask = descriptors.filters.blur.apply(
        descriptors,
        texture_pool,
        encoder,
        staging_belt,
        &input,
        blur,
        None,
    )?;
    Ok(mask.map(|target| (target, [x.start, y.start])))
}

pub struct GradientGlowFilter {
    bind_group_layout: BindGroupLayout,
    pipeline_layout: PipelineLayout,
    uniform_buffer: Buffer,
    pipelines: FilterPipelines,
}

impl GradientGlowFilter {
    pub fn new(device: &Device) -> Result<Self, RenderFailure> {
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gradient filter binds"),
                entries: &[
                    texture_entry(0),
                    texture_entry(1),
                    texture_entry(3),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(UNIFORM_SIZE),
                        },
                        count: None,
                    },
                ],
            })?;
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gradient filter"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        })?;
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gradient filter parameters and colors"),
            size: UNIFORM_SIZE.get(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })?;
        Ok(Self {
            bind_group_layout,
            pipeline_layout,
            uniform_buffer,
            pipelines: Default::default(),
        })
    }

    pub fn apply(
        &self,
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        encoder: &mut CommandEncoder,
        staging_belt: &mut StagingBelt,
        source: &FilterSource,
        output_origin: [i32; 2],
        output_size: (u32, u32),
        filter: &GradientFilter,
        kind: GradientKind,
        opaque_target: bool,
        destination: Option<FilterSource<'_>>,
    ) -> Result<CommandTarget, RenderFailure> {
        let (sin, cos) = filter.angle.to_f32().sin_cos();
        let distance = filter.distance.to_f32();
        let offset = [cos * distance, sin * distance];
        let opposite_offset = [-offset[0], -offset[1]];
        let strength = filter.strength.to_f32().max(0.0);
        let blur = filter.inner_blur_filter();
        let region = |offset: [f32; 2]| {
            if strength == 0.0 || filter.colors.is_empty() || blur.impotent() {
                return None;
            }
            let passes = blur.num_passes();
            Some([
                mask_axis(
                    output_origin[0],
                    output_size.0,
                    offset[0],
                    blur.blur_x,
                    passes,
                    source.texture.width(),
                )?,
                mask_axis(
                    output_origin[1],
                    output_size.1,
                    offset[1],
                    blur.blur_y,
                    passes,
                    source.texture.height(),
                )?,
            ])
        };
        let mut mask_region = region(offset);
        let mut opposite_region = if kind == GradientKind::Bevel {
            region(opposite_offset)
        } else {
            None
        };
        if let (Some(a), Some(b)) = (&mask_region, &opposite_region)
            && let Some(shared) =
                shared_mask_region(a, b, descriptors.limits.max_texture_dimension_2d)
        {
            mask_region = Some(shared);
            opposite_region = None;
        }
        let mut render_mask = |region| {
            blurred_mask(
                descriptors,
                texture_pool,
                encoder,
                staging_belt,
                source,
                region,
                &blur,
            )
        };
        let mask = render_mask(mask_region)?;
        let opposite_mask = render_mask(opposite_region)?;
        // A lookup missing the blurred bitmap reads zero from either mask as well as the source.
        let [(mask_view, mask_point), (opposite_view, opposite_point)] = [
            (mask.as_ref(), offset),
            (opposite_mask.as_ref().or(mask.as_ref()), opposite_offset),
        ]
        .map(|(mask, offset)| {
            let (view, origin) = mask.map_or((source.view, [0, 0]), |(mask, origin)| {
                (mask.color_view(), *origin)
            });
            (
                view,
                [
                    (output_origin[0] - origin[0]) as f32 - offset[0],
                    (output_origin[1] - origin[1]) as f32 - offset[1],
                ],
            )
        });
        let format = source.texture.format();
        let samples = source.texture.sample_count();
        let pipeline = self.pipelines.get_or_init(format, samples, || {
            let shader = &descriptors.shaders.gradient_glow_filter;
            descriptors
                .persistent_device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Gradient filter"),
                    layout: Some(&self.pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some("filter__vertex_entry_point"),
                        buffers: &VERTEX_BUFFERS_DESCRIPTION_FILTERS,
                        compilation_options: Default::default(),
                    },
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: samples,
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some("main_fragment"),
                        targets: &[Some(format.into())],
                        compilation_options: Default::default(),
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        })?;
        let target = CommandTarget::new(
            descriptors,
            texture_pool,
            wgpu::Extent3d {
                width: output_size.0,
                height: output_size.1,
                depth_or_array_layers: 1,
            },
            format,
            samples,
            match destination.filter(|dst| {
                dst.point == (0, 0)
                    && dst.size == output_size
                    && (dst.view.texture().width(), dst.view.texture().height()) == output_size
                    && dst.view.texture() != source.view.texture()
                    && dst.view.texture() != mask_view.texture()
                    && dst.view.texture() != opposite_view.texture()
            }) {
                Some(dst) => {
                    RenderTargetMode::ExistingWithView(dst.view.clone(), wgpu::Color::TRANSPARENT)
                }
                None => RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            },
            encoder,
        )?;
        let uniform = GradientUniform {
            source_point: output_origin,
            mask_point,
            opposite_point,
            kind: kind as u32,
            _padding: 0,
            strength,
            mode: if filter.is_inner() {
                1
            } else if filter.is_on_top() {
                2
            } else {
                0
            },
            opaque_target: u32::from(opaque_target),
            source_mode: if kind.is_identity(filter) {
                3
            } else if filter.is_knockout() {
                2
            } else {
                u32::from(filter.flags.contains(GradientFilterFlags::COMPOSITE_SOURCE))
            },
            colors: gradient_colors(&filter.colors, kind),
        };
        staging_belt
            .write_buffer(encoder, &self.uniform_buffer, 0, UNIFORM_SIZE)?
            .copy_from_slice(bytemuck::bytes_of(&uniform));
        let group = descriptors.cached_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gradient filter"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(opposite_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        })?;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Gradient filter"),
            color_attachments: &[target.color_attachments()],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group.raw(), &[]);
        let quad = &descriptors.quad;
        pass.set_vertex_buffer(0, quad.vertex_buffer.slice(quad.vertices_filter.clone()));
        pass.set_index_buffer(quad.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(quad.indices_quad.clone(), 0, 0..1);
        ruffle_render::profile::work(|w| {
            w.render_passes += 1;
            w.filter_passes += 1;
            match kind {
                GradientKind::Glow => w.filter_kind_glow_shadow += 1,
                GradientKind::Bevel => w.filter_kind_bevel += 1,
            }
            w.draw_calls += 1;
            w.pipeline_changes += 1;
            w.material_binds += 1;
            w.pass_pixels += u64::from(target.width()) * u64::from(target.height());
        });
        Ok(target)
    }
}
