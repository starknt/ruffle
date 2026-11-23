// A byte permutation never needs a fractional RGBA8 storage conversion.
@group(0) @binding(0) var source_image: texture_2d<f32>;
@group(0) @binding(2) var<uniform> channels: vec4<u32>;
@group(0) @binding(3) var output_image: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(output_image)) {
        return;
    }
    let position = vec2<i32>(id.xy);
    let src = textureLoad(source_image, position, 0);
    let rgb = min(vec3<f32>(src[channels.x], src[channels.y], src[channels.z]), vec3<f32>(src.a));
    textureStore(output_image, position, vec4<f32>(rgb, src.a));
}
