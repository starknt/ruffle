//! Opt-in loader diagnostics. Keep only numeric identifiers, so observing a
//! resource never promotes a weak reference or changes when the GC releases it.
use super::SwfMovieData;
use gc_arena::Collect;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Write as _;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Collect, Debug)]
#[collect(require_static)]
pub(super) struct MovieLifetime(u64);

impl MovieLifetime {
    pub fn new(movie: &SwfMovieData) -> Self {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        static NEXT: AtomicU64 = AtomicU64::new(1);
        if movie.data().is_empty()
            || !*ENABLED.get_or_init(|| {
                std::env::var_os("RUFFLE_MOVIE_LIFETIMES").as_deref() == Some("1".as_ref())
            })
        {
            return Self(0);
        }
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let mut digest = DefaultHasher::new();
        movie.data().hash(&mut digest);
        let _ = writeln!(
            std::io::stderr().lock(),
            "movie_lifetime time_us={} id={id} event=create resource={:016x} bytes={}",
            timestamp(),
            digest.finish(),
            movie.data().len()
        );
        Self(id)
    }

    pub fn record(&self, action: &str) {
        if self.0 != 0 {
            let _ = writeln!(
                std::io::stderr().lock(),
                "movie_lifetime time_us={} id={} event={action}",
                timestamp(),
                self.0
            );
        }
    }
}

fn timestamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros()
}

impl Drop for MovieLifetime {
    fn drop(&mut self) {
        self.record("drop");
    }
}
