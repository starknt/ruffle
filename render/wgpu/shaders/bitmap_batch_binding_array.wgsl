@group(2) @binding(0) var bitmaps: binding_array<texture_2d<f32>, 16>;

fn sample_bitmap(index: u32, uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(bitmaps[index], texture_sampler, uv, 0.0);
}
