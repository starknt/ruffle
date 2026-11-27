struct Convolution {
    source_point: vec2<i32>,
    source_size: vec2<i32>,
    default_color: vec4<f32>,
    divisor: f32,
    bias: f32,
    clamp_edges: u32,
    alpha_mode: u32,
    unpremultiply_factors: array<vec4<u32>, 64>,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var coefficients: texture_2d<f32>;
@group(0) @binding(2) var<uniform> filter_args: Convolution;

fn sample_color(point: vec2<i32>) -> vec4<f32> {
    let edge = filter_args.source_size - vec2<i32>(1);
    let bounded = clamp(point, vec2<i32>(0), edge);
    if filter_args.clamp_edges == 0u && any(point != bounded) {
        return filter_args.default_color;
    }
    let pixel = vec4<u32>(round(textureLoad(source, filter_args.source_point + bounded, 0) * 255.0));
    let factor = filter_args.unpremultiply_factors[pixel.a / 4u][pixel.a % 4u];
    let straight = min((pixel.rgb * factor + vec3<u32>(32768u)) >> vec3<u32>(16u), vec3<u32>(255u));
    return vec4<f32>(vec4<u32>(straight, pixel.a));
}

@fragment
fn main_fragment(in: filter__VertexOutput) -> @location(0) vec4<f32> {
    let point = vec2<i32>(in.position.xy);
    let size = vec2<i32>(textureDimensions(coefficients, 0));
    let first = point - size / 2;
    var sum = vec4<f32>(0.0);
    for (var y = 0; y < size.y; y++) {
        for (var x = 0; x < size.x; x++) {
            let tap = vec2<i32>(x, y);
            let weight = textureLoad(coefficients, tap, 0).r;
            if weight != 0.0 {
                sum += sample_color(first + tap) * weight;
            }
        }
    }
    // Flash truncates the filtered channel in byte space before premultiplying it.
    var color = floor(clamp(sum / filter_args.divisor + filter_args.bias, vec4<f32>(0.0), vec4<f32>(255.0))) / 255.0;
    if filter_args.alpha_mode == 2u {
        color.a = 1.0;
    } else if filter_args.alpha_mode == 1u {
        color.a = textureLoad(source, filter_args.source_point + point, 0).a;
    }
    return vec4<f32>(color.rgb * color.a, color.a);
}
