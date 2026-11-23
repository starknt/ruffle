fn main() {
    cfg_aliases::cfg_aliases! {
        dx12: { all(target_os = "windows", feature = "dx12") },
        gles: { feature = "gles" },
        metal: { all(target_vendor = "apple", feature = "metal") },
        vulkan: { feature = "vulkan" },
        drm: { all(
            feature = "drm",
            any(target_os = "linux", target_os = "freebsd", target_os = "netbsd", target_os = "openbsd")
        ) },
        any_backend: { any(dx12, metal, vulkan, gles) },
        // ⚠️ Keep in sync with target.cfg() definition in Cargo.toml and cfg_alias in `wgpu` crate ⚠️
        static_dxc: { all(target_os = "windows", feature = "static-dxc", not(target_arch = "aarch64"), target_env = "msvc") },
        supports_64bit_atomics: { target_has_atomic = "64" },
        supports_ptr_atomics: { target_has_atomic = "ptr" }
    }
}
