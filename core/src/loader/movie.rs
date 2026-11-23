//! Preparation owns transport bytes. Only the finished movie or non-SWF body crosses back to the player.
use super::{ContentType, Error, Instant, LoadBytesInfo, SwfMovieData, extract_swz};

pub(super) enum PreparedContent {
    Movie(SwfMovieData),
    Other(Vec<u8>),
}

impl PreparedContent {
    pub(super) fn into_movie(self) -> Result<SwfMovieData, Error> {
        match self {
            Self::Movie(movie) => Ok(movie),
            Self::Other(bytes) => Err(Error::UnexpectedData(
                ContentType::Swf,
                ContentType::sniff(&bytes),
            )),
        }
    }
}

use std::sync::{Arc, LazyLock};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

// Acquire before polling a native network fetch, which may buffer the whole response.
const MAX_PREPARATIONS: usize = 4;
static PREPARATIONS: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(MAX_PREPARATIONS)));

pub(super) enum Preparation {
    Inline,
    Worker(OwnedSemaphorePermit),
}

impl Preparation {
    pub(super) async fn for_url(url: &str) -> Self {
        if use_inline_movie_parse(url) {
            return Self::Inline;
        }
        Self::Worker(
            PREPARATIONS
                .clone()
                .acquire_owned()
                .await
                .expect("permanent movie pool"),
        )
    }

    // Flash supplies loadBytes as an immediate snapshot. If all workers are busy,
    // prepare it on the next player-thread poll outside the player lock.
    pub(super) fn for_bytes() -> Self {
        PREPARATIONS
            .clone()
            .try_acquire_owned()
            .map_or(Self::Inline, Self::Worker)
    }

    pub(super) async fn prepare(
        self,
        body: Vec<u8>,
        url: String,
        loader_url: Option<String>,
        info: Option<LoadBytesInfo>,
    ) -> (Result<PreparedContent, swf::error::Error>, u64) {
        let start = Instant::now();
        let work = move || prepare_swf_movie_data(body, url, loader_url, info);
        let result = match self {
            Self::Inline => work(),
            Self::Worker(permit) => run_worker(permit, work).await,
        };
        (result, start.elapsed().as_nanos() as u64)
    }
}

async fn run_worker<T: Send + 'static>(
    permit: OwnedSemaphorePermit,
    work: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (sender, receiver) = futures::channel::oneshot::channel();
    rayon::spawn_fifo(move || {
        let _permit = permit;
        if !sender.is_canceled() {
            let _ = sender.send(work());
        }
    });
    receiver.await.expect("movie worker dropped result")
}

#[cfg(test)]
pub(super) async fn prepare_swf_movie(
    body: Vec<u8>,
    url: String,
    loader_url: Option<String>,
    info: Option<LoadBytesInfo>,
) -> Result<PreparedContent, swf::error::Error> {
    Preparation::for_url(&url)
        .await
        .prepare(body, url, loader_url, info)
        .await
        .0
}

/// Fully prepare immutable movie data, including the SWZ envelope and shared
/// backing file. Callers install this result without parsing or copying it again.
pub(super) fn prepare_swf_movie_data(
    body: Vec<u8>,
    url: String,
    loader_url: Option<String>,
    load_bytes_info: Option<LoadBytesInfo>,
) -> Result<PreparedContent, swf::error::Error> {
    let transport = body;
    let uncompressed = if ContentType::sniff(&transport) == ContentType::Swf {
        None
    } else {
        extract_swz(&transport).ok()
    };
    let parse_body = uncompressed.as_deref().unwrap_or(&transport);
    if ContentType::sniff(parse_body) != ContentType::Swf {
        return Ok(PreparedContent::Other(transport));
    }
    let movie = SwfMovieData::from_data(parse_body, url, loader_url, load_bytes_info)?
        .with_shared_bytes(true)
        .with_compressed_len(transport.len());
    // The movie records the original transport length for progress/bytesTotal;
    // do not send the transport buffer back through the loader channel.
    Ok(PreparedContent::Movie(movie))
}

pub(super) fn use_inline_movie_parse(url: &str) -> bool {
    url.starts_with("file:") || url.starts_with("data:") || url.starts_with("ruffle:")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{FutureExt, task::noop_waker};
    use std::{
        future::Future,
        sync::atomic::{AtomicBool, Ordering},
        task::Context,
    };

    #[test]
    fn movie_preparation_limits_admission_and_releases_cancelled_waiters() {
        futures::executor::block_on(async {
            let mut permits = Vec::new();
            for _ in 0..MAX_PREPARATIONS {
                permits.push(Preparation::for_url("https://example.test/movie.swf").await);
            }
            assert!(
                Preparation::for_url("https://example.test/obsolete.swf")
                    .now_or_never()
                    .is_none()
            );
            assert!(matches!(Preparation::for_bytes(), Preparation::Inline));
            permits.pop();
            let replacement = Preparation::for_url("https://example.test/replacement.swf").await;
            assert!(matches!(replacement, Preparation::Worker(_)));
        });
    }

    #[test]
    fn queued_worker_skips_a_cancelled_movie_before_parsing() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let permit = Arc::new(Semaphore::new(1)).try_acquire_owned().unwrap();
        let parsed = Arc::new(AtomicBool::new(false));
        let observed = parsed.clone();
        let (done, finished) = std::sync::mpsc::channel();
        pool.install(move || {
            let mut pending = Box::pin(run_worker(permit, move || {
                observed.store(true, Ordering::SeqCst)
            }));
            assert!(
                pending
                    .as_mut()
                    .poll(&mut Context::from_waker(&noop_waker()))
                    .is_pending()
            );
            drop(pending);
            rayon::spawn_fifo(move || done.send(()).unwrap());
        });
        finished
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert!(!parsed.load(Ordering::SeqCst));
    }
}
