//! Installation-private cache files shared by the native asset and GPU caches.
#[cfg(windows)]
#[path = "windows_acl.rs"]
mod windows_acl;
use std::fs::{self, File, Metadata};
use std::io::{self, ErrorKind};
#[cfg(unix)]
use std::os::unix::fs::DirBuilderExt;
use std::path::{Component, Path, PathBuf};

pub fn private_dir(dir: &Path) -> io::Result<PathBuf> {
    if dir.components().any(|c| c == Component::ParentDir) {
        return Err(ErrorKind::InvalidInput.into());
    }
    // A trailing slash forces Unix lstat to follow even a final directory symlink.
    let dir: PathBuf = std::path::absolute(dir)?.components().collect();
    let parent = dir.parent().ok_or(ErrorKind::InvalidInput)?;
    check_ancestry(parent)?;
    let builder = fs::DirBuilder::new();
    #[cfg(unix)]
    let mut builder = builder;
    #[cfg(unix)]
    builder.mode(0o700);
    match builder.create(&dir) {
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
        result => result?,
    }
    check_private_dir(&dir)?;
    // A private directory can still be unwritable due to ACLs or a read-only mount.
    private_tempfile(&dir)?.close()?;
    Ok(dir)
}

pub fn private_tempfile(dir: &Path) -> io::Result<tempfile::NamedTempFile> {
    let tmp = tempfile::NamedTempFile::new_in(dir)?;
    #[cfg(windows)]
    windows_acl::check_handle(tmp.as_file(), false)?;
    Ok(tmp)
}

pub fn open_private_read(path: &Path) -> io::Result<File> {
    check_file_metadata(&fs::symlink_metadata(path)?)?;
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
    check_file_metadata(&metadata)?;
    #[cfg(windows)]
    windows_acl::check_handle(&file, false)?;
    Ok(file)
}

pub fn private_lock(path: &Path) -> io::Result<File> {
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
    check_file_metadata(&f.metadata()?)?;
    #[cfg(windows)]
    windows_acl::check_handle(&f, false)?;
    Ok(f)
}

pub fn check_ancestry(path: &Path) -> io::Result<()> {
    for ancestor in path.ancestors() {
        let metadata = fs::symlink_metadata(ancestor)?;
        if !metadata.is_dir() || is_link(&metadata) {
            return Err(ErrorKind::PermissionDenied.into());
        }
        #[cfg(windows)]
        windows_acl::check_path(ancestor, true)?;
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
pub fn android_private_group(uid: u32, gid: u32) -> bool {
    (10_000..20_000).contains(&(uid % 100_000))
        && (gid == uid || gid.checked_sub(10_000) == Some(uid))
}

pub fn check_private_dir(path: &Path) -> io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() {
        return Err(ErrorKind::PermissionDenied.into());
    }
    #[cfg(unix)]
    if std::os::unix::fs::MetadataExt::mode(&metadata) & 0o300 != 0o300 {
        return Err(ErrorKind::PermissionDenied.into());
    }
    #[cfg(windows)]
    windows_acl::check_path(path, false)?;
    check_private(&metadata)
}

pub fn check_private_file(_path: &Path, metadata: &Metadata) -> io::Result<()> {
    check_file_metadata(metadata)?;
    #[cfg(windows)]
    windows_acl::check_path(_path, false)?;
    Ok(())
}

fn check_file_metadata(metadata: &Metadata) -> io::Result<()> {
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
