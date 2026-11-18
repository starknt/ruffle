// NOTE: The `shader_filter_common.wgsl` source is prepended to this before compilation.

struct Filter {
    r_to_r: f32,
    g_to_r: f32,
    b_to_r: f32,
    a_to_r: f32,
    r_extra: f32,

    r_to_g: f32,
    g_to_g: f32,
    b_to_g: f32,
    a_to_g: f32,
    g_extra: f32,

    r_to_b: f32,
    g_to_b: f32,
    b_to_b: f32,
    a_to_b: f32,
    b_extra: f32,

    r_to_a: f32,
    g_to_a: f32,
    b_to_a: f32,
    a_to_a: f32,
    a_extra: f32,
}

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
struct MatrixChain {
    channels: vec4<u32>,
    matrix: Filter,
}
@group(0) @binding(2) var<uniform> filter_args: MatrixChain;

@vertex
fn main_vertex(in: filter__VertexInput) -> filter__VertexOutput {
    return filter__main_vertex(in);
}

// Keep the original per-component arithmetic and premultiplication order.
fn apply_matrix(src: vec4<f32>, f: Filter) -> vec4<f32> {
    var color = vec4<f32>(
        clamp((f.r_to_r * src.r / src.a) + (f.g_to_r * src.g / src.a) + (f.b_to_r * src.b / src.a) + (f.a_to_r * src.a) + (f.r_extra / 255.0), 0.0, 1.0),
        clamp((f.r_to_g * src.r / src.a) + (f.g_to_g * src.g / src.a) + (f.b_to_g * src.b / src.a) + (f.a_to_g * src.a) + (f.g_extra / 255.0), 0.0, 1.0),
        clamp((f.r_to_b * src.r / src.a) + (f.g_to_b * src.g / src.a) + (f.b_to_b * src.b / src.a) + (f.a_to_b * src.a) + (f.b_extra / 255.0), 0.0, 1.0),
        clamp((f.r_to_a * src.r / src.a) + (f.g_to_a * src.g / src.a) + (f.b_to_a * src.b / src.a) + (f.a_to_a * src.a) + (f.a_extra / 255.0), 0.0, 1.0),
    );
    return vec4<f32>(color.rgb * color.a, color.a);
}

@fragment
fn main_fragment(in: filter__VertexOutput) -> @location(0) vec4<f32> {
    let src = textureSample(texture, texture_sampler, in.uv);
    // Intermediate channel permutations are exactly representable in RGBA8.
    // All other matrices keep separate native attachments and their quantization.
    let channels = filter_args.channels;
    return apply_matrix(vec4<f32>(src[channels.x], src[channels.y], src[channels.z], src.a), filter_args.matrix);
}
