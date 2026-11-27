use ruffle_render_wgpu::gpu_resources::Device;
use std::fs::DirBuilder;
use std::hash::{DefaultHasher, Hash, Hasher};

// Bump when standalone player shaders or pipeline layouts change.
const HOST_ABI: &str = "ruffle-desktop-shaders-layouts-v2";

pub(super) fn configure(device: &Device) {
    let (Some(cache), Ok(exe)) = (dirs::cache_dir(), std::env::current_exe()) else {
        return;
    };
    let parent = cache.join("ruffle-desktop-pipelines");
    let mut builder = DirBuilder::new();
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    if let Err(error) = builder.recursive(true).create(&parent) {
        tracing::debug!(%error, "Native pipeline cache directory unavailable");
        return;
    }
    let mut installation = DefaultHasher::new();
    exe.hash(&mut installation);
    let dir = parent.join(format!("{:016x}", installation.finish()));
    // SAFETY: This installation's directory is reserved for native snapshots under the
    // OS per-user cache (inheriting the profile ACL on Windows). The engine checks the
    // path and private files before loading. Neither movies nor CLI options select it.
    let status = unsafe { device.enable_pipeline_cache(&dir, HOST_ABI) };
    tracing::info!(?status, "Native pipeline cache");
}
