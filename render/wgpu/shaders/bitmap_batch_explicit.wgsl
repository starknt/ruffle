@group(2) @binding(0) var bitmap_0: texture_2d<f32>;
@group(2) @binding(1) var bitmap_1: texture_2d<f32>;
@group(2) @binding(2) var bitmap_2: texture_2d<f32>;
@group(2) @binding(3) var bitmap_3: texture_2d<f32>;
@group(2) @binding(4) var bitmap_4: texture_2d<f32>;
@group(2) @binding(5) var bitmap_5: texture_2d<f32>;
@group(2) @binding(6) var bitmap_6: texture_2d<f32>;
@group(2) @binding(7) var bitmap_7: texture_2d<f32>;
@group(2) @binding(8) var bitmap_8: texture_2d<f32>;
@group(2) @binding(9) var bitmap_9: texture_2d<f32>;
@group(2) @binding(10) var bitmap_10: texture_2d<f32>;
@group(2) @binding(11) var bitmap_11: texture_2d<f32>;
@group(2) @binding(12) var bitmap_12: texture_2d<f32>;
@group(2) @binding(13) var bitmap_13: texture_2d<f32>;
@group(2) @binding(14) var bitmap_14: texture_2d<f32>;
@group(2) @binding(15) var bitmap_15: texture_2d<f32>;

fn sample_bitmap(index: u32, uv: vec2<f32>) -> vec4<f32> {
    var color: vec4<f32>;
    // Every input has one mip. An explicit LOD permits per-instance selection
    // without derivative uniformity requirements or changing the sampler.
    switch index {
        case 0u: { color = textureSampleLevel(bitmap_0, texture_sampler, uv, 0.0); }
        case 1u: { color = textureSampleLevel(bitmap_1, texture_sampler, uv, 0.0); }
        case 2u: { color = textureSampleLevel(bitmap_2, texture_sampler, uv, 0.0); }
        case 3u: { color = textureSampleLevel(bitmap_3, texture_sampler, uv, 0.0); }
        case 4u: { color = textureSampleLevel(bitmap_4, texture_sampler, uv, 0.0); }
        case 5u: { color = textureSampleLevel(bitmap_5, texture_sampler, uv, 0.0); }
        case 6u: { color = textureSampleLevel(bitmap_6, texture_sampler, uv, 0.0); }
        case 7u: { color = textureSampleLevel(bitmap_7, texture_sampler, uv, 0.0); }
        case 8u: { color = textureSampleLevel(bitmap_8, texture_sampler, uv, 0.0); }
        case 9u: { color = textureSampleLevel(bitmap_9, texture_sampler, uv, 0.0); }
        case 10u: { color = textureSampleLevel(bitmap_10, texture_sampler, uv, 0.0); }
        case 11u: { color = textureSampleLevel(bitmap_11, texture_sampler, uv, 0.0); }
        case 12u: { color = textureSampleLevel(bitmap_12, texture_sampler, uv, 0.0); }
        case 13u: { color = textureSampleLevel(bitmap_13, texture_sampler, uv, 0.0); }
        case 14u: { color = textureSampleLevel(bitmap_14, texture_sampler, uv, 0.0); }
        default: { color = textureSampleLevel(bitmap_15, texture_sampler, uv, 0.0); }
    }
    return color;
}
