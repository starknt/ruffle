//! Installation-private, bounded derived byte streams. Each class owns a directory and a one-item save queue.
use super::{CacheFile, HEADER_LEN, Hash, check_private_file, private_dir, private_lock};
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
};
pub(crate) struct Files {
    pub(super) dir: PathBuf,
    writer: Option<mpsc::SyncSender<(CacheFile, Vec<u8>)>>,
    worker: Option<thread::JoinHandle<()>>,
    hits: AtomicU64,
    misses: AtomicU64,
}
impl Files {
    pub fn open(dir: &Path, name: &str, switch: &str, budget: u64) -> Option<Self> {
        if std::env::var(switch).as_deref() != Ok("1") {
            return None;
        }
        let dir = private_dir(&private_dir(dir).ok()?.join(name)).ok()?;
        let (writer, receiver) = mpsc::sync_channel::<(CacheFile, Vec<u8>)>(1);
        let worker = thread::Builder::new()
            .name("ruffle-derived-save".into())
            .spawn(move || {
                for (file, payload) in receiver {
                    let _ = write(&file, &payload, budget);
                }
            })
            .ok()?;
        Some(Self {
            dir,
            writer: Some(writer),
            worker: Some(worker),
            hits: 0.into(),
            misses: 0.into(),
        })
    }
    pub fn stats(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    pub(super) fn file(&self, identity: Hash, extension: &str) -> CacheFile {
        CacheFile {
            path: self.dir.join(format!("{identity:x}.{extension}")),
            identity,
        }
    }
    pub fn note(&self, hit: bool) {
        if hit { &self.hits } else { &self.misses }.fetch_add(1, Ordering::Relaxed);
    }
    pub(super) fn store(&self, file: CacheFile, payload: Vec<u8>) {
        if let Some(tx) = &self.writer {
            let _ = tx.try_send((file, payload));
        }
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        self.writer.take();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}
pub(super) fn write(file: &CacheFile, payload: &[u8], budget: u64) -> io::Result<()> {
    let dir = file.checked_dir()?;
    let lock = private_lock(&dir.join("budget.lock"))?;
    lock.try_lock().map_err(io::Error::from)?;
    let mut used = payload.len() as u64 + HEADER_LEN as u64;
    if used > budget {
        return Err(io::ErrorKind::InvalidData.into());
    }
    let mut old = vec![];
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path == file.path
            || path.extension() != file.path.extension()
            || !path
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            continue;
        }
        let meta = fs::symlink_metadata(&path)?;
        if check_private_file(&path, &meta).is_err() {
            continue;
        }
        used = used.saturating_add(meta.len());
        old.push((meta.modified()?, meta.len(), path));
    }
    old.sort_unstable();
    for (_, len, path) in old {
        if used <= budget {
            break;
        }
        fs::remove_file(path)?;
        used = used.saturating_sub(len);
    }
    file.store(payload)
}
