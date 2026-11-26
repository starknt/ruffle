//! Installation-private storage for native wgpu pipeline cache snapshots.

use futures::FutureExt;
use sha2::{Digest, Sha256, digest::Output};
use std::fs::{self, File, Metadata};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Component, Path, PathBuf};

const MAGIC: &[u8; 8] = b"RFLWGPU1";
// Bump when engine shaders or pipeline layouts change independently of the crate version.
const ENGINE_ABI: &str = "ruffle-shaders-layouts-v1";
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
    cache: Option<(wgpu::PipelineCache, CacheFile)>,
}

impl PipelineCacheState {
    /// # Safety
    /// The host must reserve `dir` for this installation and control its ancestry,
    /// ACLs and contents across runs. Only this writer may supply snapshot bytes.
    pub(crate) unsafe fn open(device: &wgpu::Device, dir: &Path, host_abi: &str) -> Self {
        let inactive = |status| Self {
            status,
            cache: None,
        };
        if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return inactive(PipelineCacheStatus::Unsupported);
        }
        let Some(key) = identity(&device.adapter_info(), host_abi) else {
            return inactive(PipelineCacheStatus::Unsupported);
        };
        let Ok(file) = CacheFile::new(dir, key) else {
            return inactive(PipelineCacheStatus::Unavailable);
        };
        match file.load() {
            Ok(data) => {
                // SAFETY: the host guarantees provenance; load verified this whole snapshot.
                if let Some(cache) = unsafe { create_cache(device, Some(&data)) } {
                    return Self {
                        status: PipelineCacheStatus::Loaded,
                        cache: Some((cache, file)),
                    };
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
            Some(cache) => Self {
                status: PipelineCacheStatus::Cold,
                cache: Some((cache, file)),
            },
            None => inactive(PipelineCacheStatus::Unavailable),
        }
    }

    pub(crate) fn handle(&self) -> Option<&wgpu::PipelineCache> {
        self.cache.as_ref().map(|(cache, _)| cache)
    }

    pub(crate) fn status(&self) -> PipelineCacheStatus {
        self.status
    }

    pub(crate) fn flush(&self) -> bool {
        let Some((cache, file)) = &self.cache else {
            return false;
        };
        cache
            .get_data()
            .is_some_and(|data| file.store(&data).is_ok())
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

#[derive(Debug)]
struct CacheFile {
    path: PathBuf,
    identity: Hash,
}

impl CacheFile {
    fn new(dir: &Path, identity: Hash) -> io::Result<Self> {
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
        self.checked_dir()?;
        check_private_file(&fs::symlink_metadata(&self.path)?)?;
        #[cfg(unix)]
        let mut file = {
            use rustix::fs::{Mode, OFlags, open};
            File::from(open(
                &self.path,
                OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
                Mode::empty(),
            )?)
        };
        #[cfg(not(unix))]
        let mut file = {
            let mut options = File::options();
            #[cfg(windows)]
            // FILE_FLAG_OPEN_REPARSE_POINT lets us validate the opened object itself.
            std::os::windows::fs::OpenOptionsExt::custom_flags(&mut options, 0x0020_0000);
            options.read(true).open(&self.path)?
        };
        let metadata = file.metadata()?;
        check_private_file(&metadata)?;
        let mut header = [0; HEADER_LEN];
        file.read_exact(&mut header)?;
        let len = u64::from_le_bytes(header[40..48].try_into().expect("eight-byte length field"));
        if &header[..8] != MAGIC
            || header[8..40] != self.identity[..]
            || !(1..=MAX_SNAPSHOT).contains(&len)
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
        assert!(cold.flush());
        assert_eq!(open(&device, "v1").status(), PipelineCacheStatus::Loaded);
        assert_eq!(open(&device, "v2").status(), PipelineCacheStatus::Cold);
        let path = fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path();
        fs::write(path, b"interrupted snapshot").unwrap();
        let rebuilt = open(&device, "v1");
        assert_eq!(rebuilt.status(), PipelineCacheStatus::Cold);
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
