//! Installation-private storage for native wgpu pipeline cache snapshots.

pub(crate) mod decoded;
pub(crate) mod derived;
pub(crate) mod mesh;
mod writer;
use futures::FutureExt;
use sha2::{Digest, Sha256, digest::Output};
use std::fs::{self, File, Metadata};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Component, Path, PathBuf};
use writer::Writer;

const MAGIC: &[u8; 8] = b"RFLWGPU1";
// Bump when engine shaders or pipeline layouts change independently of the crate version.
pub(crate) const ENGINE_ABI: &str = "ruffle-shaders-layouts-v3";
const MAX_SNAPSHOT: u64 = 64 * 1024 * 1024;
const HEADER_LEN: usize = 8 + 32 + 8 + 32;
type Hash = Output<Sha256>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineCacheStatus {
    Disabled,
    Unsupported,
    Unavailable,
    Cold,
    /// wgpu accepted a local snapshot. Drivers do not report per-pipeline hits.
    Loaded,
}

#[derive(Debug)]
pub(crate) struct PipelineCacheState {
    status: PipelineCacheStatus,
    writer: Option<Writer>,
    cache: Option<wgpu::PipelineCache>,
    _active: Option<File>,
}

impl PipelineCacheState {
    /// # Safety
    /// The host must reserve `dir` for this installation and control its ancestry,
    /// ACLs and contents across runs. Only this writer may supply snapshot bytes.
    pub(crate) unsafe fn open(device: &wgpu::Device, dir: &Path, host_abi: &str) -> Self {
        let inactive = |status| Self {
            status,
            writer: None,
            cache: None,
            _active: None,
        };
        if std::env::var_os("RUFFLE_PIPELINE_CACHE").as_deref() == Some(std::ffi::OsStr::new("0")) {
            return inactive(PipelineCacheStatus::Disabled);
        }
        if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return inactive(PipelineCacheStatus::Unsupported);
        }
        let Some(key) = identity(&device.adapter_info(), host_abi) else {
            return inactive(PipelineCacheStatus::Unsupported);
        };
        let Ok(file) = CacheFile::new(dir, key) else {
            return inactive(PipelineCacheStatus::Unavailable);
        };
        let Ok(active) = private_lock(&file.path.with_extension("active")).and_then(|f| {
            f.lock_shared()?;
            Ok(f)
        }) else {
            return inactive(PipelineCacheStatus::Unavailable);
        };
        let complete = |status, cache: wgpu::PipelineCache| {
            let writer = private_lock(&file.path.with_extension("writer"))
                .ok()
                .and_then(|lock| {
                    lock.try_lock().ok()?;
                    let handle = cache.clone();
                    let file = file.clone();
                    Writer::new(move || {
                        let _writer = &lock;
                        handle
                            .get_data()
                            .is_some_and(|data| file.store_budgeted(&data, SNAPSHOT_BUDGET).is_ok())
                    })
                    .ok()
                });
            Self {
                status,
                cache: Some(cache),
                writer,
                _active: Some(active),
            }
        };
        match file.load() {
            Ok(data) => {
                // SAFETY: the host guarantees provenance; load verified this whole snapshot.
                if let Some(cache) = unsafe { create_cache(device, Some(&data)) } {
                    return complete(PipelineCacheStatus::Loaded, cache);
                }
            }
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::NotFound | ErrorKind::InvalidData | ErrorKind::UnexpectedEof
                ) => {}
            Err(_) => return inactive(PipelineCacheStatus::Unavailable),
        }
        // SAFETY: creating an empty cache supplies no driver data.
        match unsafe { create_cache(device, None) } {
            Some(cache) => complete(PipelineCacheStatus::Cold, cache),
            None => inactive(PipelineCacheStatus::Unavailable),
        }
    }

    pub(crate) fn handle(&self) -> Option<&wgpu::PipelineCache> {
        self.cache.as_ref()
    }

    pub(crate) fn status(&self) -> PipelineCacheStatus {
        self.status
    }

    pub(crate) fn flush(&self) -> bool {
        self.writer.as_ref().is_some_and(|w| w.save(true))
    }

    pub(crate) fn dirty(&self) {
        if let Some(w) = &self.writer {
            w.dirty()
        }
    }
    pub(crate) fn schedule(&self) {
        if let Some(w) = &self.writer {
            w.save(false);
        }
    }
}

/// `data`, if supplied, must be a complete, authentic output of `get_data`.
unsafe fn create_cache(device: &wgpu::Device, data: Option<&[u8]>) -> Option<wgpu::PipelineCache> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    // SAFETY: the caller guarantees the provenance of data.
    let cache = unsafe {
        device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
            label: Some("Ruffle native pipeline cache"),
            data,
            fallback: false,
        })
    };
    // Native wgpu-core returns ready futures. Pop every scope in LIFO order.
    let errors = [validation, internal, oom].map(|scope| scope.pop().now_or_never());
    // Device loss bypasses error scopes and leaves an unreadable native handle.
    (errors.iter().all(|e| matches!(e, Some(None))) && cache.get_data().is_some()).then_some(cache)
}

fn identity(info: &wgpu::AdapterInfo, host_abi: &str) -> Option<Hash> {
    let adapter_key = wgpu::util::pipeline_cache_key(info)?;
    let mut hash = Sha256::new();
    for field in [
        adapter_key.as_str(),
        &info.name,
        &info.device_pci_bus_id,
        &info.driver,
        &info.driver_info,
        std::env::consts::OS,
        std::env::consts::ARCH,
        env!("CARGO_PKG_VERSION"),
        ENGINE_ABI,
        host_abi,
    ] {
        hash.update((field.len() as u64).to_le_bytes());
        hash.update(field.as_bytes());
    }
    Some(hash.finalize())
}

#[derive(Clone, Debug)]
struct CacheFile {
    path: PathBuf,
    identity: Hash,
}

pub(crate) fn private_dir(dir: &Path) -> io::Result<PathBuf> {
    if dir.components().any(|c| c == Component::ParentDir) {
        return Err(ErrorKind::InvalidInput.into());
    }
    // A trailing slash forces Unix lstat to follow even a final directory symlink.
    let dir: PathBuf = std::path::absolute(dir)?.components().collect();
    let parent = dir.parent().ok_or(ErrorKind::InvalidInput)?;
    check_ancestry(parent)?;
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    match builder.create(&dir) {
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
        result => result?,
    }
    check_private_dir(&dir)?;
    // A private directory can still be unwritable due to ACLs or a read-only mount.
    tempfile::NamedTempFile::new_in(&dir)?.close()?;
    Ok(dir)
}

impl CacheFile {
    fn new(dir: &Path, identity: Hash) -> io::Result<Self> {
        let dir = private_dir(dir)?;
        Ok(Self {
            path: dir.join(format!("{identity:x}.bin")),
            identity,
        })
    }

    fn checked_dir(&self) -> io::Result<&Path> {
        let dir = self.path.parent().ok_or(ErrorKind::InvalidInput)?;
        check_ancestry(dir.parent().ok_or(ErrorKind::InvalidInput)?)?;
        check_private_dir(dir)?;
        Ok(dir)
    }

    fn load(&self) -> io::Result<Vec<u8>> {
        self.load_bounded(MAX_SNAPSHOT)
    }
    fn load_bounded(&self, limit: u64) -> io::Result<Vec<u8>> {
        self.checked_dir()?;
        let mut file = open_private_read(&self.path)?;
        let metadata = file.metadata()?;
        let mut header = [0; HEADER_LEN];
        file.read_exact(&mut header)?;
        let len = u64::from_le_bytes(header[40..48].try_into().expect("eight-byte length field"));
        if &header[..8] != MAGIC
            || header[8..40] != self.identity[..]
            || !(1..=limit.min(MAX_SNAPSHOT)).contains(&len)
            || metadata.len() != HEADER_LEN as u64 + len
        {
            return Err(ErrorKind::InvalidData.into());
        }
        let mut data = Vec::new();
        data.try_reserve_exact(len as usize)
            .map_err(io::Error::other)?;
        data.resize(len as usize, 0);
        file.read_exact(&mut data)?;
        if file.read(&mut [0])? != 0 || header[48..] != self.seal(&data)[..] {
            return Err(ErrorKind::InvalidData.into());
        }
        Ok(data)
    }

    fn seal(&self, data: &[u8]) -> Hash {
        Sha256::new()
            .chain_update(MAGIC)
            .chain_update(self.identity)
            .chain_update((data.len() as u64).to_le_bytes())
            .chain_update(data)
            .finalize()
    }

    fn store_budgeted(&self, data: &[u8], budget: u64) -> io::Result<()> {
        let len = data.len() as u64;
        if !(1..=MAX_SNAPSHOT).contains(&len) || len + HEADER_LEN as u64 > budget {
            return Err(ErrorKind::InvalidData.into());
        }
        let dir = self.checked_dir()?;
        let directory_lock = private_lock(&dir.join("budget.lock"))?;
        directory_lock.lock()?;
        let mut used = len + HEADER_LEN as u64;
        let mut older = Vec::new();
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if path == self.path || path.extension().and_then(|v| v.to_str()) != Some("bin") {
                continue;
            }
            let Some(stem) = path
                .file_stem()
                .and_then(|v| v.to_str())
                .filter(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()))
            else {
                continue;
            };
            let identity: Option<Vec<u8>> = (0..32)
                .map(|i| u8::from_str_radix(&stem[2 * i..2 * i + 2], 16).ok())
                .collect();
            let Some(identity) = identity else { continue };
            let metadata = fs::symlink_metadata(&path)?;
            if check_private_file(&metadata).is_err() {
                continue;
            }
            used = used.saturating_add(metadata.len());
            older.push((
                metadata.modified()?,
                metadata.len(),
                Self {
                    path,
                    identity: Hash::from_slice(&identity).to_owned(),
                },
            ));
        }
        older.sort_by_key(|v| v.0);
        for (_, size, old) in older {
            if used <= budget {
                break;
            }
            let active = private_lock(&old.path.with_extension("active"))?;
            // An idle file in our private snapshot namespace is disposable even if
            // its payload is corrupt. Recheck the file after taking its active lock.
            if active.try_lock().is_ok() && open_private_read(&old.path).is_ok() {
                fs::remove_file(&old.path)?;
                used = used.saturating_sub(size);
            }
        }
        if used > budget {
            return Err(io::Error::other("snapshot directory budget exhausted"));
        }
        self.store(data)
    }

    fn store(&self, data: &[u8]) -> io::Result<()> {
        let len = data.len() as u64;
        if !(1..=MAX_SNAPSHOT).contains(&len) {
            return Err(ErrorKind::InvalidData.into());
        }
        let dir = self.checked_dir()?;
        match fs::symlink_metadata(&self.path) {
            Ok(metadata) => check_private_file(&metadata)?,
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        let mut temp = tempfile::NamedTempFile::new_in(dir)?;
        temp.write_all(MAGIC)?;
        temp.write_all(&self.identity)?;
        temp.write_all(&len.to_le_bytes())?;
        temp.write_all(&self.seal(data))?;
        temp.write_all(data)?;
        temp.as_file().sync_all()?;
        temp.persist(&self.path)?;
        Ok(())
    }
}

const SNAPSHOT_BUDGET: u64 = if cfg!(target_os = "android") {
    128
} else {
    256
} * 1024
    * 1024;

// Stable lock files are never unlinked: all processes lock the same inode across replacement.
pub(crate) fn open_private_read(path: &Path) -> io::Result<File> {
    check_private_file(&fs::symlink_metadata(path)?)?;
    #[cfg(unix)]
    let file = {
        use rustix::fs::{Mode, OFlags, open};
        File::from(open(
            path,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )?)
    };
    #[cfg(not(unix))]
    let file = {
        let mut options = File::options();
        #[cfg(windows)]
        // FILE_FLAG_OPEN_REPARSE_POINT lets us validate the opened object itself.
        std::os::windows::fs::OpenOptionsExt::custom_flags(&mut options, 0x0020_0000);
        options.read(true).open(path)?
    };
    let metadata = file.metadata()?;
    check_private_file(&metadata)?;
    Ok(file)
}

fn private_lock(path: &Path) -> io::Result<File> {
    let mut options = File::options();
    options.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(
            (rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::NONBLOCK).bits() as i32,
        );
    }
    #[cfg(windows)]
    std::os::windows::fs::OpenOptionsExt::custom_flags(&mut options, 0x0020_0000);
    let f = options.open(path)?;
    check_private_file(&f.metadata()?)?;
    Ok(f)
}

fn check_ancestry(path: &Path) -> io::Result<()> {
    for ancestor in path.ancestors() {
        let metadata = fs::symlink_metadata(ancestor)?;
        if !metadata.is_dir() || is_link(&metadata) {
            return Err(ErrorKind::PermissionDenied.into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            // Android's system account (AID_SYSTEM) manages the app data hierarchy.
            let system = |id| id == 0 || (cfg!(target_os = "android") && id == 1000);
            let uid = rustix::process::geteuid().as_raw();
            let trusted_owner = system(metadata.uid()) || metadata.uid() == uid;
            let mode = metadata.mode();
            let gid = metadata.gid();
            let trusted_group =
                cfg!(target_os = "android") && (system(gid) || android_private_group(uid, gid));
            let group_writable = mode & 0o020 != 0 && !trusted_group;
            if !trusted_owner || (mode & 0o1000 == 0 && (mode & 0o002 != 0 || group_writable)) {
                return Err(ErrorKind::PermissionDenied.into());
            }
        }
    }
    Ok(())
}

// Android assigns each app a private GID and a cache GID in the same user profile.
#[cfg(any(unix, test))]
fn android_private_group(uid: u32, gid: u32) -> bool {
    (10_000..20_000).contains(&(uid % 100_000))
        && (gid == uid || gid.checked_sub(10_000) == Some(uid))
}

fn check_private_dir(path: &Path) -> io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() {
        return Err(ErrorKind::PermissionDenied.into());
    }
    #[cfg(unix)]
    if std::os::unix::fs::MetadataExt::mode(&metadata) & 0o300 != 0o300 {
        return Err(ErrorKind::PermissionDenied.into());
    }
    check_private(&metadata)
}

fn check_private_file(metadata: &Metadata) -> io::Result<()> {
    if !metadata.is_file() {
        return Err(ErrorKind::PermissionDenied.into());
    }
    #[cfg(unix)]
    if std::os::unix::fs::MetadataExt::nlink(metadata) > 1 {
        return Err(ErrorKind::PermissionDenied.into());
    }
    check_private(metadata)
}

fn check_private(metadata: &Metadata) -> io::Result<()> {
    if is_link(metadata) {
        return Err(ErrorKind::PermissionDenied.into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.uid() != rustix::process::geteuid().as_raw() || metadata.mode() & 0o077 != 0 {
            return Err(ErrorKind::PermissionDenied.into());
        }
    }
    Ok(())
}

fn is_link(metadata: &Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        // FILE_ATTRIBUTE_REPARSE_POINT includes junctions as well as symbolic links.
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.is_symlink()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn private_test_root() -> tempfile::TempDir {
        tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap()
    }

    #[test]
    fn cache_budget_evicts_only_idle_complete_snapshots() {
        let root = private_test_root();
        let dir = root.path().join("gpu");
        let files: Vec<_> = (0..3)
            .map(|i| CacheFile::new(&dir, [i; 32].into()).unwrap())
            .collect();
        let bytes = [0; 128];
        let budget = 2 * (HEADER_LEN as u64 + 128);
        files[0].store_budgeted(&bytes, budget).unwrap();
        files[1].store_budgeted(&bytes, budget).unwrap();
        let active = private_lock(&files[0].path.with_extension("active")).unwrap();
        active.lock_shared().unwrap();
        fs::write(dir.join("unrelated.bin"), [1]).unwrap();
        files[2].store_budgeted(&bytes, budget).unwrap();
        assert!(files[0].load().is_ok());
        assert!(!files[1].path.exists());
        assert_eq!(files[2].load().unwrap(), bytes);
        assert!(dir.join("unrelated.bin").exists());
    }

    #[test]
    fn corrupt_inactive_snapshot_cannot_exhaust_another_keys_budget() {
        let root = private_test_root();
        let dir = root.path().join("gpu");
        let old = CacheFile::new(&dir, [10; 32].into()).unwrap();
        let new = CacheFile::new(&dir, [11; 32].into()).unwrap();
        let budget = 1024;
        for size in [3, MAX_SNAPSHOT + 4096] {
            old.store(b"previous driver").unwrap();
            File::options()
                .write(true)
                .open(&old.path)
                .unwrap()
                .set_len(size)
                .unwrap();
            let active = private_lock(&old.path.with_extension("active")).unwrap();
            active.lock_shared().unwrap();
            let payload = vec![7; (budget - HEADER_LEN as u64) as usize];
            assert!(new.store_budgeted(&payload, budget).is_err());
            assert!(old.path.exists());
            drop(active);
            new.store_budgeted(&payload, budget).unwrap();
            assert!(!old.path.exists());
            assert_eq!(new.load().unwrap(), payload);
        }
    }

    #[test]
    fn independent_process_cannot_replace_an_active_writer() {
        const KEY: &str = "RUFFLE_TEST_PIPELINE_WRITER_FILE";
        if let Some(path) = std::env::var_os(KEY) {
            let file = private_lock(Path::new(&path)).unwrap();
            assert!(matches!(
                file.try_lock(),
                Err(std::fs::TryLockError::WouldBlock)
            ));
            return;
        }
        let root = private_test_root();
        let path = root.path().join("writer");
        let file = private_lock(&path).unwrap();
        file.try_lock().unwrap();
        assert!(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "pipeline_cache::tests::independent_process_cannot_replace_an_active_writer"
                ])
                .env(KEY, &path)
                .status()
                .unwrap()
                .success()
        );
        drop(file);
        assert!(private_lock(&path).unwrap().try_lock().is_ok());
    }

    #[test]
    fn android_cache_groups_belong_to_one_user_and_app() {
        assert!(android_private_group(10_123, 10_123));
        assert!(android_private_group(10_123, 20_123));
        assert!(android_private_group(110_123, 120_123));
        assert!(!android_private_group(110_123, 20_123));
        assert!(!android_private_group(10_123, 20_124));
        assert!(!android_private_group(20_123, 30_123));
    }

    #[test]
    fn only_the_complete_snapshot_for_this_identity_round_trips() {
        let root = private_test_root();
        let file = CacheFile::new(&root.path().join("gpu"), [1; 32].into()).unwrap();
        file.store(b"first").unwrap();
        file.store(b"second snapshot").unwrap();
        assert_eq!(file.load().unwrap(), b"second snapshot");
        assert!(file.store(b"").is_err());
        assert_eq!(file.load().unwrap(), b"second snapshot");

        let sealed = std::fs::read(&file.path).unwrap();
        let other = CacheFile::new(&root.path().join("gpu"), [2; 32].into()).unwrap();
        std::fs::copy(&file.path, &other.path).unwrap();
        assert!(other.load().is_err());
        for offset in [0, 8, 40, 48, sealed.len() - 1] {
            let mut damaged = sealed.clone();
            damaged[offset] ^= 1;
            std::fs::write(&file.path, damaged).unwrap();
            assert!(file.load().is_err());
        }
        for length in [0, 7, sealed.len() - 1, sealed.len() + 1] {
            let mut damaged = sealed.clone();
            damaged.resize(length, 0);
            std::fs::write(&file.path, damaged).unwrap();
            assert!(file.load().is_err());
        }
        let mut huge = sealed;
        huge[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&file.path, huge).unwrap();
        assert!(file.load().is_err());
    }

    #[test]
    fn readers_see_complete_snapshots_during_concurrent_replacement() {
        let root = private_test_root();
        let file = CacheFile::new(&root.path().join("gpu"), [0; 32].into()).unwrap();
        let first = vec![0x11; 8192];
        let second = vec![0x22; 16384];
        file.store(&first).unwrap();
        let start = std::sync::Barrier::new(3);
        std::thread::scope(|scope| {
            for bytes in [&first, &second] {
                let (file, start) = (&file, &start);
                scope.spawn(move || {
                    start.wait();
                    for _ in 0..12 {
                        file.store(bytes).unwrap();
                    }
                });
            }
            start.wait();
            for _ in 0..48 {
                let snapshot = file.load().unwrap();
                assert!(snapshot == first || snapshot == second);
            }
        });
    }

    #[test]
    fn key_changes_with_the_adapter_driver_and_host_abi() {
        let mut info =
            wgpu::AdapterInfo::new(wgpu::DeviceType::IntegratedGpu, wgpu::Backend::Vulkan);
        let original = identity(&info, "abi1").unwrap();
        assert_ne!(identity(&info, "abi2").unwrap(), original);
        info.device += 1;
        assert_ne!(identity(&info, "abi1").unwrap(), original);
        info.device -= 1;
        info.driver_info.push_str("updated");
        assert_ne!(identity(&info, "abi1").unwrap(), original);
        info.backend = wgpu::Backend::Gl;
        assert!(identity(&info, "abi1").is_none());
    }

    #[test]
    #[ignore = "requires a Vulkan adapter with PIPELINE_CACHE"]
    fn native_snapshot_reopens_and_corruption_rebuilds() {
        use futures::executor::block_on;
        let root = private_test_root();
        let dir = root.path().join("gpu");
        let mut options = wgpu::InstanceDescriptor::new_without_display_handle();
        options.backends = wgpu::Backends::VULKAN;
        let adapter =
            block_on(wgpu::Instance::new(options).request_adapter(&Default::default())).unwrap();
        let (disabled, _) = block_on(adapter.request_device(&Default::default())).unwrap();
        // SAFETY: the test owns the fresh, dedicated storage across all opens.
        let open = |device, abi| unsafe { PipelineCacheState::open(device, &dir, abi) };
        let unsupported = open(&disabled, "v1");
        assert_eq!(unsupported.status(), PipelineCacheStatus::Unsupported);
        assert!(!unsupported.flush());
        assert!(!dir.exists());
        let (device, _) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::PIPELINE_CACHE,
            ..Default::default()
        }))
        .unwrap();
        let cold = open(&device, "v1");
        assert_eq!(cold.status(), PipelineCacheStatus::Cold);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
        });
        let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: cold.handle(),
        });
        cold.dirty();
        assert!(cold.flush());
        drop(cold);
        assert_eq!(open(&device, "v1").status(), PipelineCacheStatus::Loaded);
        assert_eq!(open(&device, "v2").status(), PipelineCacheStatus::Cold);
        let path = CacheFile::new(&dir, identity(&device.adapter_info(), "v1").unwrap())
            .unwrap()
            .path;
        fs::write(path, b"interrupted snapshot").unwrap();
        let rebuilt = open(&device, "v1");
        assert_eq!(rebuilt.status(), PipelineCacheStatus::Cold);
        rebuilt.dirty();
        assert!(rebuilt.flush());
        assert_eq!(open(&device, "v1").status(), PipelineCacheStatus::Loaded);
        device.destroy();
        for abi in ["v1", "v2"] {
            let lost = open(&device, abi);
            assert_eq!(lost.status(), PipelineCacheStatus::Unavailable);
            assert!(lost.handle().is_none());
            assert!(!lost.flush());
        }
    }

    #[cfg(unix)]
    #[test]
    fn unsafe_paths_are_never_read_or_replaced() {
        use std::fs;
        use std::os::unix::fs::{MetadataExt, PermissionsExt, symlink};
        let root = private_test_root();
        let dir = root.path().join("gpu");
        let file = CacheFile::new(&dir, [0; 32].into()).unwrap();
        file.store(b"private").unwrap();
        assert_eq!(dir.metadata().unwrap().mode() & 0o777, 0o700);
        assert_eq!(file.path.metadata().unwrap().mode() & 0o777, 0o600);
        let elsewhere = root.path().join("elsewhere");
        fs::rename(&file.path, &elsewhere).unwrap();
        symlink(&elsewhere, &file.path).unwrap();
        assert!(file.load().is_err());
        assert!(file.store(b"replacement").is_err());
        assert!(file.path.is_symlink());
        fs::remove_file(&file.path).unwrap();
        fs::hard_link(&elsewhere, &file.path).unwrap();
        assert!(file.load().is_err());
        fs::remove_file(&file.path).unwrap();
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            rustix::fs::mkfifoat(rustix::fs::CWD, &file.path, rustix::fs::Mode::RUSR).unwrap();
            assert!(file.load().is_err());
            assert!(file.store(b"replacement").is_err());
            fs::remove_file(&file.path).unwrap();
        }
        fs::rename(&elsewhere, &file.path).unwrap();
        fs::set_permissions(&file.path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(file.load().is_err());
        assert!(file.store(b"replacement").is_err());
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(CacheFile::new(&dir, [0; 32].into()).is_err());
        assert!(file.store(b"replacement").is_err());
        let link = root.path().join("link");
        symlink(&dir, &link).unwrap();
        assert!(CacheFile::new(&link, [0; 32].into()).is_err());
        assert!(CacheFile::new(&link.join("nested"), [0; 32].into()).is_err());
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o500)).unwrap();
        assert!(CacheFile::new(&dir, [0; 32].into()).is_err());
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
        let mut trailing = link.as_os_str().to_owned();
        trailing.push("/");
        assert!(CacheFile::new(Path::new(&trailing), [0; 32].into()).is_err());
        assert!(CacheFile::new(&dir.join(".."), [0; 32].into()).is_err());
        assert!(!dir.join("nested").exists());
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o777)).unwrap();
        assert!(CacheFile::new(&dir.join("nested"), [0; 32].into()).is_err());
        assert!(!dir.join("nested").exists());
    }
}
