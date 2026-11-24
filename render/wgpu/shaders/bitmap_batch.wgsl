// Fixed bindings require no descriptor-indexing feature. Only single-mip normal
// bitmap quads with one sampler are admitted; arithmetic matches bitmap.wgsl.
// Keep 16 slots in sync with layouts::BITMAP_BATCH_TEXTURES.
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) mult_color: vec4<f32>,
    @location(2) add_color: vec4<f32>,
    @location(3) @interpolate(flat) texture_index: u32,
};
@group(1) @binding(0) var<uniform> transforms: array<common__Transforms, max_transforms>;
@group(2) @binding(16) var texture_sampler: sampler;
@vertex
fn main_vertex(in: common__VertexInputUv, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
    let pos = common__globals.global_matrix * (transforms[instanceIndex].world_matrix * vec4<f32>(in.position.x, in.position.y, 0.0, 1.0));
    return VertexOutput(pos, in.uv.xy / in.uv.z, transforms[instanceIndex].mult_color, transforms[instanceIndex].add_color, instanceIndex % 16u);
}
@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = sample_bitmap(in.texture_index, in.uv);
    if (color.a > 0.0) {
        color = vec4<f32>(color.rgb / color.a, color.a);
        color = color * in.mult_color + in.add_color;
        color = saturate(color);
        color = vec4<f32>(color.rgb * color.a, color.a);
    }
    return color;
}
