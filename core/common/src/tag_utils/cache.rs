//! Verified FWS mappings. The caller grants disk eligibility only after fetching the original bytes.
use super::*;
use ruffle_memory::private_cache::{
    check_ancestry, check_private_dir, check_private_file, open_private_read, private_dir,
    private_lock,
};
use std::{
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

const PREFIX: u64 = 72;
const MAGIC: &[u8; 8] = b"RFLSWF01";
const MAX_FILE: usize = 64 << 20;
const BUDGET: u64 = 256 << 20;

#[derive(Clone, Debug)]
pub struct MovieCache {
    directory: PathBuf,
    budget: u64,
}

impl MovieCache {
    pub fn open(directory: &Path) -> io::Result<Self> {
        Self::open_with_budget(directory, BUDGET)
    }
    pub fn open_with_budget(directory: &Path, budget: u64) -> io::Result<Self> {
        let cache = Self {
            directory: private_dir(directory)?,
            budget: budget.min(BUDGET),
        };
        let lock = private_lock(&cache.directory.join("budget.lock"))?;
        lock.lock()?;
        cache.trim(None, 0)?;
        Ok(cache)
    }

    fn check_dir(dir: &Path) -> io::Result<()> {
        check_ancestry(dir.parent().ok_or(io::ErrorKind::InvalidInput)?)?;
        check_private_dir(dir)
    }
    fn entries(dir: &Path) -> io::Result<Vec<(std::time::SystemTime, u64, PathBuf)>> {
        let mut entries = Vec::new();
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("fws")
                || !path
                    .file_stem()
                    .and_then(|v| v.to_str())
                    .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
            {
                continue;
            }
            let meta = fs::symlink_metadata(&path)?;
            check_private_file(&path, &meta)?;
            entries.push((meta.modified()?, meta.len(), path));
        }
        Ok(entries)
    }

    pub fn usage(dir: &Path) -> io::Result<(u64, u64)> {
        Self::check_dir(dir)?;
        let lock = private_lock(&dir.join("budget.lock"))?;
        lock.lock_shared()?;
        let entries = Self::entries(dir)?;
        Ok((
            entries.len() as u64,
            entries.iter().map(|(_, bytes, _)| bytes).sum(),
        ))
    }

    pub fn clear(dir: &Path) -> io::Result<()> {
        Self::check_dir(dir)?;
        let lock = private_lock(&dir.join("budget.lock"))?;
        lock.lock()?;
        for (_, _, path) in Self::entries(dir)? {
            fs::remove_file(path)?;
        }
        Ok(())
    }

    pub fn load(
        &self,
        source: &[u8],
        url: String,
        loader_url: Option<String>,
    ) -> Result<SwfMovieData, swf::error::Error> {
        let length = source
            .get(4..8)
            .map(|s| u32::from_le_bytes(s.try_into().unwrap()) as usize)
            .unwrap_or(0);
        if !(64 * 1024..=MAX_FILE).contains(&length)
            || PREFIX + length as u64 > self.budget
            || source.get(3) == Some(&0)
        {
            return SwfMovieData::from_data(source, url, loader_url, None)
                .map(|m| m.with_shared_bytes(true));
        }
        let key = ContentId::from_parts(&[MAGIC, source]);
        let name: String = key.0.iter().map(|b| format!("{b:02x}")).collect();
        let path = self.directory.join(name).with_extension("fws");
        if let Some((header, backing)) = self.read(&path, source, length, key) {
            return Ok(SwfMovieData::from_backing(
                header,
                backing,
                url,
                loader_url,
                None,
                source.len(),
            ));
        }
        let movie = SwfMovieData::from_data(source, url, loader_url, None)?.with_shared_bytes(true);
        if let Some(file) = movie.shared_file()
            && file.file().len() == length
        {
            let _ = self.write(&path, key, file.file());
        }
        Ok(movie)
    }

    fn read(
        &self,
        path: &Path,
        source: &[u8],
        length: usize,
        key: ContentId,
    ) -> Option<(HeaderExt, Arc<MovieBytes>)> {
        Self::check_dir(&self.directory).ok()?;
        let mut file = open_private_read(path).ok()?;
        if file.metadata().ok()?.len() != PREFIX + length as u64 {
            return None;
        }
        let mut header = [0; PREFIX as usize];
        file.read_exact(&mut header).ok()?;
        if &header[..8] != MAGIC || header[8..40] != key.0 {
            return None;
        }
        // Writers publish by atomic replacement and never modify an existing mapping.
        let view = unsafe {
            memmap2::MmapOptions::new()
                .offset(PREFIX)
                .len(length)
                .map(&file)
                .ok()?
        };
        let checksum = ContentId::from_parts(&[&key.0, &view]);
        if header[40..] != checksum.0 || view.get(..3)? != b"FWS" || view[3..8] != source[3..8] {
            return None;
        }
        let compression = swf::read::read_compression_type(source).ok()?;
        let (header, offset) =
            swf::read::read_movie_header(&view[8..], source[3], compression, length as u32).ok()?;
        let tags_offset = offset + 8;
        let id = ContentId::from_parts(&[&view[..tags_offset], &view[tags_offset..]]);
        let scope = MemoryScope::current().manager().shared();
        let bytes = scope.manager().resources::<MovieBytes>().intern(id, || {
            let charge = scope.track(MemoryKind::MappedFile, length as u64);
            Arc::new(MovieBytes {
                file: Vec::new(),
                tags_offset,
                charge: scope.track(MemoryKind::MovieBytes, 0),
                mapping: Some(Arc::new(Mapping {
                    file,
                    view,
                    offset: PREFIX,
                    _charge: charge,
                })),
            })
        });
        Some((header, bytes))
    }

    // Callers hold the directory budget lock while selecting and removing entries.
    fn trim(&self, replaced: Option<&Path>, incoming: u64) -> io::Result<()> {
        let mut entries = Self::entries(&self.directory)?;
        entries.retain(|(_, _, old)| Some(old.as_path()) != replaced);
        let mut used = entries
            .iter()
            .fold(incoming, |sum, (_, bytes, _)| sum.saturating_add(*bytes));
        entries.sort_unstable();
        for (_, len, old) in entries {
            if used <= self.budget {
                break;
            }
            fs::remove_file(old)?;
            used -= len;
        }
        Ok(())
    }

    fn write(&self, path: &Path, key: ContentId, bytes: &[u8]) -> io::Result<()> {
        Self::check_dir(&self.directory)?;
        let incoming = PREFIX + bytes.len() as u64;
        if incoming > self.budget {
            return Err(io::ErrorKind::InvalidData.into());
        }
        let lock = private_lock(&self.directory.join("budget.lock"))?;
        lock.try_lock().map_err(io::Error::from)?;
        self.trim(Some(path), incoming)?;
        let mut temp = ruffle_memory::private_cache::private_tempfile(&self.directory)?;
        temp.write_all(MAGIC)?;
        temp.write_all(&key.0)?;
        temp.write_all(&ContentId::from_parts(&[&key.0, bytes]).0)?;
        temp.write_all(bytes)?;
        temp.persist(path)?;
        Ok(())
    }
}
