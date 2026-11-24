// Each invocation owns one material run of at most 16 ordered unit quads.
// Equal bindings, stencil and render-target state were established by the CPU.
struct Transform {
    world: mat4x4<f32>,
    multiply: vec4<f32>,
    add: vec4<f32>,
}
struct IndexedDraw {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}
struct Result {
    args: array<IndexedDraw, MAX_DRAWS>,
    count: u32,
}
@group(0) @binding(0) var<storage, read> runs: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> results: array<Result>;
@group(0) @binding(2) var<uniform> transforms: array<Transform, MAX_TRANSFORMS>;
@group(0) @binding(3) var<uniform> projection: mat4x4<f32>;

fn potentially_visible(index: u32) -> bool {
    let corners = array(vec2f(0,0), vec2f(1,0), vec2f(1,1), vec2f(0,1));
    var outside = vec4<bool>(true);
    for (var i = 0u; i < 4u; i++) {
        let p = projection * (transforms[index].world * vec4f(corners[i], 0, 1));
        outside = vec4<bool>(
            outside.x && (p.x < -p.w), outside.y && (p.x > p.w),
            outside.z && (p.y < -p.w), outside.w && (p.y > p.w));
    }
    return !any(outside);
}

@compute @workgroup_size(1)
fn prepare(@builtin(global_invocation_id) id: vec3<u32>) {
    let run = runs[id.x];
    if MAX_DRAWS == 1u {
        results[id.x].args[0] = IndexedDraw(run.y, run.w, run.x, 0, run.z);
        results[id.x].count = 1u;
        return;
    }
    // Consecutive visible instances can still share one draw. First-instance
    // indices continue selecting the same transforms and textures after gaps.
    var count = 0u;
    var span = 0u;
    for (var i = 0u; i <= run.w; i++) {
        if i < run.w && potentially_visible(run.z + i) {
            span++;
        } else if span != 0u {
            results[id.x].args[count] = IndexedDraw(run.y, span, run.x, 0, run.z + i - span);
            count++;
            span = 0u;
        }
    }
    results[id.x].count = count;
    for (var i = count; i < MAX_DRAWS; i++) {
        results[id.x].args[i] = IndexedDraw(0u, 0u, 0u, 0, 0u);
    }
}
