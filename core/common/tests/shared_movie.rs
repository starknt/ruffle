use ruffle_common::tag_utils::SwfMovieData;
use std::sync::Arc;

#[test]
fn identical_movies_share_the_immutable_file_across_independent_loads() {
    let bytes = include_bytes!("../../../tests/tests/swfs/native_perf/cache_soak/child.swf");
    let load = |url: &str| {
        SwfMovieData::from_data(bytes, url.to_owned(), None, None)
            .unwrap()
            .with_shared_bytes(true)
    };
    let a = load("file:///world/one.swf");
    let b = load("file:///world/two.swf");
    assert_ne!(a.url(), b.url());
    assert!(Arc::ptr_eq(
        &a.shared_file().unwrap(),
        &b.shared_file().unwrap()
    ));
}

#[test]
fn corrupt_disk_entry_rebuilds_and_mapped_loader_bytes_are_private() {
    use ruffle_common::tag_utils::MovieCache;
    let original = include_bytes!("../../../tests/tests/swfs/native_perf/cache_soak/child.swf");
    let data = SwfMovieData::from_data(original, "file:///seed.swf".into(), None, None)
        .unwrap()
        .with_shared_bytes(true);
    let mut source = data.shared_file().unwrap().file().to_vec();
    let len = 96 * 1024_u32;
    source.resize(len as usize, 0);
    source[4..8].copy_from_slice(&len.to_le_bytes());
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("cache");
    let cache = MovieCache::open(&dir).unwrap();
    let load = || {
        cache
            .load(&source, "file:///large.swf".into(), None)
            .unwrap()
    };
    drop(load());
    let entry = dir
        .read_dir()
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|s| s == "fws"))
        .unwrap();
    let expected = std::fs::read(&entry).unwrap();
    let mut corrupt = expected.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    std::fs::write(&entry, corrupt).unwrap();
    drop(load());
    assert_eq!(std::fs::read(&entry).unwrap(), expected);
    let file = load().shared_file().unwrap();
    assert_eq!(file.file(), source);
    let mut view = file.private_pages().unwrap();
    assert_eq!(view.len(), source.len());
    view[10] ^= 1;
    assert_eq!(file.file()[10], source[10]);
    assert_eq!(std::fs::read(entry).unwrap(), expected);
}

#[test]
fn lowering_the_disk_budget_reclaims_old_movies_while_live_views_remain_readable() {
    use ruffle_common::tag_utils::MovieCache;
    let original = include_bytes!("../../../tests/tests/swfs/native_perf/cache_soak/child.swf");
    let movie = SwfMovieData::from_data(original, "file:///seed.swf".into(), None, None)
        .unwrap()
        .with_shared_bytes(true);
    let mut source = movie.shared_file().unwrap().file().to_vec();
    let len = 96 * 1024_u32;
    source.resize(len as usize, 0);
    source[4..8].copy_from_slice(&len.to_le_bytes());
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("cache");
    let cache = MovieCache::open(&dir).unwrap();
    drop(
        cache
            .load(&source, "file:///large.swf".into(), None)
            .unwrap(),
    );
    let mapped = cache
        .load(&source, "file:///large.swf".into(), None)
        .unwrap();
    assert_eq!(MovieCache::usage(&dir).unwrap().0, 1);
    drop(cache);
    let smaller = MovieCache::open_with_budget(&dir, 16 * 1024).unwrap();
    assert_eq!(MovieCache::usage(&dir).unwrap(), (0, 0));
    assert_eq!(mapped.shared_file().unwrap().file(), source);
    drop(
        smaller
            .load(&source, "file:///other.swf".into(), None)
            .unwrap(),
    );
    assert_eq!(MovieCache::usage(&dir).unwrap(), (0, 0));
}
