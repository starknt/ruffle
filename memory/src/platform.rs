pub(super) fn cpu_budget() -> u64 {
    let fallback = if cfg!(target_os = "android") {
        512 << 20
    } else {
        2 << 30
    };
    effective_memory().map_or(fallback, |bytes| (bytes / 4).max(1))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn effective_memory() -> Option<u64> {
    use std::{fs, path::Path};
    let mut limit = fs::read_to_string("/proc/meminfo").ok().and_then(|text| {
        text.lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()?
            .checked_mul(1024)
    });
    let groups = fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    for line in groups.lines() {
        let mut fields = line.splitn(3, ':');
        let (_, Some(controllers), Some(relative)) = (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let (root, file) = if controllers.is_empty() {
            (Path::new("/sys/fs/cgroup"), "memory.max")
        } else if controllers.split(',').any(|name| name == "memory") {
            (Path::new("/sys/fs/cgroup/memory"), "memory.limit_in_bytes")
        } else {
            continue;
        };
        let path = root.join(relative.trim_start_matches('/'));
        for path in path.ancestors().take_while(|path| path.starts_with(root)) {
            if let Some(value) = fs::read_to_string(path.join(file))
                .ok()
                .and_then(|text| text.trim().parse::<u64>().ok())
            {
                limit = Some(limit.map_or(value, |old| old.min(value)));
            }
        }
    }
    limit
}

#[cfg(target_os = "macos")]
fn effective_memory() -> Option<u64> {
    let mut bytes = 0u64;
    let mut length = size_of_val(&bytes);
    // SAFETY: the output points at a writable u64, with its actual size.
    let status = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && length == size_of_val(&bytes)).then_some(bytes)
}

#[cfg(windows)]
fn effective_memory() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: GlobalMemoryStatusEx fills the initialized output structure synchronously.
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status.ullTotalPhys)
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    windows
)))]
fn effective_memory() -> Option<u64> {
    None
}
