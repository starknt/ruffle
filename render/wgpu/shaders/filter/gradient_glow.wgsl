struct GradientFilter {
    source_point: vec2<i32>,
    mask_point: vec2<f32>,
    opposite_point: vec2<f32>,
    strength: f32,
    mode: u32,
    opaque_target: u32,
    source_mode: u32, // 0 hides the source, 1 composites it, 2 knocks it out, 3 copies it.
    kind: u32, // GradientKind: 0 for glow, 1 for bevel.
    // Uniform arrays have a 16-byte stride. Each vector holds four RGBA8 colors.
    colors: array<vec4<u32>, 64>,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var mask: texture_2d<f32>;
@group(0) @binding(2) var<uniform> gradient: GradientFilter;
@group(0) @binding(3) var opposite_mask: texture_2d<f32>;

fn mask_alpha(tex: texture_2d<f32>, pixel: vec2<i32>) -> f32 {
    if (any(pixel < vec2<i32>(0)) || any(pixel >= vec2<i32>(textureDimensions(tex)))) {
        return 0.0;
    }
    let alpha = textureLoad(tex, pixel, 0).a;
    // Keep bevel samples in byte space through interpolation so exact half steps survive floor().
    return select(alpha, round(alpha * 255.0), gradient.kind == 1u);
}

fn shifted_alpha(tex: texture_2d<f32>, point: vec2<f32>) -> f32 {
    let base = vec2<i32>(floor(point));
    let weight = fract(point);
    let a = mask_alpha(tex, base);
    if (all(weight == vec2<f32>(0.0))) {
        return a;
    }
    let b = mask_alpha(tex, base + vec2<i32>(1, 0));
    let c = mask_alpha(tex, base + vec2<i32>(0, 1));
    let d = mask_alpha(tex, base + vec2<i32>(1, 1));
    return mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
}

fn palette_index(pixel: vec2<f32>) -> u32 {
    if (gradient.strength == 0.0) {
        return 128u;
    }
    let alpha = shifted_alpha(mask, gradient.mask_point + pixel);
    if (gradient.kind == 1u) {
        let opposite = shifted_alpha(opposite_mask, gradient.opposite_point + pixel);
        let delta = clamp((opposite - alpha) * gradient.strength, -255.0, 255.0);
        // Flash halves the signed 8-bit alpha difference, rounding toward the lower ratio.
        return u32(floor(128.0 + delta * 0.5));
    }
    return u32(round(saturate(alpha * gradient.strength) * 255.0));
}

fn composite(src: vec4<f32>, pixel: vec2<i32>) -> vec4<f32> {
    if (gradient.source_mode == 3u) {
        return src;
    }
    let ratio = palette_index(vec2<f32>(pixel));
    let stop = unpack4x8unorm(gradient.colors[ratio / 4u][ratio % 4u]);
    let color = vec4<f32>(stop.rgb * stop.a, stop.a);
    let keep_source = gradient.source_mode == 1u;
    switch gradient.mode {
        case 1u: { // Inner
            let inside = color * src.a;
            return select(inside, inside + src * (1.0 - color.a), keep_source);
        }
        case 2u: { // Full
            return select(color, color + src * (1.0 - color.a), keep_source);
        }
        default: { // Outer
            let outside = color * (1.0 - src.a);
            if (gradient.source_mode == 2u) {
                return outside;
            }
            return select(color, src + outside, keep_source);
        }
    }
}

@fragment
fn main_fragment(in: filter__VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    let source_pixel = gradient.source_point + pixel;
    var src = vec4<f32>(0.0);
    if (all(source_pixel >= vec2<i32>(0)) && all(source_pixel < vec2<i32>(textureDimensions(source)))) {
        src = textureLoad(source, source_pixel, 0);
    }
    let result = composite(src, pixel);
    if (gradient.opaque_target != 0u) {
        return vec4<f32>(result.rgb / max(result.a, 1e-10), 1.0);
    }
    return result;
}
