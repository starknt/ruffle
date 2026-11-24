@group(2) @binding(0) var bitmaps: texture_2d_array<f32>;

fn sample_bitmap(index: u32, uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(bitmaps, texture_sampler, uv, i32(index), 0.0);
}
