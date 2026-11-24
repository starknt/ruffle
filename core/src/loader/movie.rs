//! Preparation owns transport bytes. Only the finished movie or non-SWF body crosses back to the player.
mod snapshot;
use super::{ContentType, Error, Instant, LoadBytesInfo, SwfMovieData, extract_swz};

pub(super) enum PreparedContent {
    Movie(SwfMovieData),
    Other(bytes::Bytes),
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

const PAGE_BYTES: usize = 64 * 1024;

#[derive(Clone)]
struct ByteWindow {
    permits: Arc<Semaphore>,
    pages: u32,
}

impl ByteWindow {
    fn new(pages: u32) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(pages as usize)),
            pages,
        }
    }

    // Unknown and oversize bodies have exclusive access. An individual valid
    // movie can exceed the ordinary shared window without deadlocking itself.
    fn units(&self, bytes: Option<usize>) -> u32 {
        bytes.map_or(self.pages as usize, |n| {
            n.div_ceil(PAGE_BYTES).clamp(1, self.pages as usize)
        }) as u32
    }

    async fn reserve(&self, bytes: Option<usize>) -> OwnedSemaphorePermit {
        self.permits
            .clone()
            .acquire_many_owned(self.units(bytes))
            .await
            .expect("permanent movie window")
    }
}

#[derive(Clone)]
pub(super) struct MovieBudgets {
    input: ByteWindow,
    decoded: ByteWindow,
    transfers: Arc<Semaphore>,
    spill: snapshot::SnapshotWriter,
}

impl Default for MovieBudgets {
    fn default() -> Self {
        Self::new(1024, 2048)
    }
}

pub(super) struct MovieInput {
    pub bytes: bytes::Bytes,
    _reservation: OwnedSemaphorePermit,
}

impl MovieBudgets {
    fn new(input_pages: u32, decoded_pages: u32) -> Self {
        Self {
            input: ByteWindow::new(input_pages),
            decoded: ByteWindow::new(decoded_pages),
            spill: Default::default(),
            transfers: Arc::new(Semaphore::new(MAX_PREPARATIONS)),
        }
    }

    #[cfg(test)]
    async fn snapshot(&self, bytes: bytes::Bytes) -> MovieInput {
        let reservation = self.input.reserve(Some(bytes.len())).await;
        MovieInput {
            bytes,
            _reservation: reservation,
        }
    }

    pub(super) async fn reserve_before_fetch(&self, address: &str) -> Option<OwnedSemaphorePermit> {
        // The native file navigator reads the complete file before yielding
        // its response. Network headers and budgeted asset-cache hits may proceed
        // concurrently, then reserve their movie input before consuming the body.
        if address.starts_with("file:") {
            Some(self.input.reserve(None).await)
        } else {
            None
        }
    }

    pub(super) async fn read(
        &self,
        response: Box<dyn super::SuccessResponse>,
        reservation: Option<OwnedSemaphorePermit>,
    ) -> Result<MovieInput, Error> {
        let mut reservation = match reservation {
            Some(reservation) => reservation,
            None => self.input.reserve(response.buffered_body_len()).await,
        };
        let bytes = response.body().await?;
        let held = reservation
            .split(self.input.units(Some(bytes.len())) as usize)
            .ok_or_else(|| Error::FetchError("Movie body exceeded its buffered length".into()))?;
        Ok(MovieInput {
            bytes: bytes.into(),
            _reservation: held,
        })
    }

    pub(super) async fn fetch(
        &self,
        address: &str,
        fetch: super::OwnedFuture<Box<dyn super::SuccessResponse>, super::ErrorResponse>,
    ) -> Result<(MovieInput, String, u16, bool), super::ErrorResponse> {
        let reservation = self.reserve_before_fetch(address).await;
        let response = fetch.await?;
        let url = response.url().into_owned();
        let status = response.status();
        let redirected = response.redirected();
        let input =
            self.read(response, reservation)
                .await
                .map_err(|error| super::ErrorResponse {
                    url: url.clone(),
                    error,
                })?;
        Ok((input, url, status, redirected))
    }
}

pub(super) enum Preparation {
    Inline,
    Worker(OwnedSemaphorePermit),
}

impl Preparation {
    pub(super) async fn for_url(url: &str, budgets: &MovieBudgets) -> Self {
        if use_inline_movie_parse(url) {
            return Self::Inline;
        }
        Self::Worker(
            budgets
                .transfers
                .clone()
                .acquire_owned()
                .await
                .expect("permanent movie pool"),
        )
    }

    // Flash supplies loadBytes as an immediate snapshot. If all workers are busy,
    // prepare it on the next player-thread poll outside the player lock.
    pub(super) fn for_bytes(budgets: &MovieBudgets) -> Self {
        budgets
            .transfers
            .clone()
            .try_acquire_owned()
            .map_or(Self::Inline, Self::Worker)
    }

    pub(super) async fn prepare(
        self,
        input: MovieInput,
        budget: &MovieBudgets,
        url: String,
        loader_url: Option<String>,
        info: Option<LoadBytesInfo>,
    ) -> (
        Result<PreparedContent, swf::error::Error>,
        u64,
        OwnedSemaphorePermit,
    ) {
        let bytes = &input.bytes;
        let decoded = if ContentType::sniff(bytes) == ContentType::Swf && bytes.len() >= 8 {
            Some(u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize)
        } else if bytes.first() == Some(&0x30) {
            // SWZ wraps the SWF length in a DER envelope.
            None
        } else {
            Some(bytes.len())
        };
        let delivery = budget.decoded.reserve(decoded).await;
        let start = Instant::now();
        let work = move || {
            let result = prepare_swf_movie_data(input.bytes, url, loader_url, info);
            drop(input._reservation);
            (result, start.elapsed().as_nanos() as u64, delivery)
        };
        match self {
            Self::Inline => work(),
            Self::Worker(transfer) => {
                let worker = PREPARATIONS
                    .clone()
                    .acquire_owned()
                    .await
                    .expect("permanent movie pool");
                run_worker(worker, move || {
                    let _transfer = transfer;
                    work()
                })
                .await
            }
        }
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
    let budget = MovieBudgets::default();
    let input = budget.snapshot(body.into()).await;
    Preparation::for_url(&url, &budget)
        .await
        .prepare(input, &budget, url, loader_url, info)
        .await
        .0
}

/// Fully prepare immutable movie data, including the SWZ envelope and shared
/// backing file. Callers install this result without parsing or copying it again.
pub(super) fn prepare_swf_movie_data(
    body: impl Into<bytes::Bytes>,
    url: String,
    loader_url: Option<String>,
    load_bytes_info: Option<LoadBytesInfo>,
) -> Result<PreparedContent, swf::error::Error> {
    let transport = body.into();
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
            let budget = MovieBudgets::default();
            let mut permits = Vec::new();
            for _ in 0..MAX_PREPARATIONS {
                permits.push(Preparation::for_url("https://example.test/movie.swf", &budget).await);
            }
            assert!(
                Preparation::for_url("https://example.test/obsolete.swf", &budget)
                    .now_or_never()
                    .is_none()
            );
            assert!(matches!(
                Preparation::for_bytes(&budget),
                Preparation::Inline
            ));
            assert!(
                Preparation::for_url("https://example.test/sibling", &MovieBudgets::default())
                    .now_or_never()
                    .is_some()
            );
            permits.pop();
            let replacement =
                Preparation::for_url("https://example.test/replacement.swf", &budget).await;
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
    #[test]
    fn a_stalled_vm_holds_its_movies_budget_through_delivery_and_cancellation() {
        let budget = MovieBudgets::new(2, 2);
        let other = MovieBudgets::new(2, 2);
        let bytes = bytes::Bytes::from(vec![7; 3 * PAGE_BYTES]);
        futures::executor::block_on(async {
            let input = budget.snapshot(bytes.clone()).await;
            assert!(budget.snapshot(bytes.clone()).now_or_never().is_none());
            let (result, _, delivery) = Preparation::Inline
                .prepare(input, &budget, String::new(), None, None)
                .await;
            assert!(matches!(result, Ok(PreparedContent::Other(_))));
            let input = budget.snapshot(bytes.clone()).await;
            let next = Preparation::Inline.prepare(input, &budget, String::new(), None, None);
            futures::pin_mut!(next);
            assert!(next.as_mut().now_or_never().is_none());
            let independent = other.snapshot(bytes.clone()).await;
            assert!(
                Preparation::Inline
                    .prepare(independent, &other, String::new(), None, None)
                    .await
                    .0
                    .is_ok()
            );
            drop((result, delivery));
            assert!(next.await.0.is_ok());
            let cancelled = budget.snapshot(bytes.clone()).await;
            drop(cancelled);
            assert!(budget.snapshot(bytes).now_or_never().is_some());
        });
    }
    #[test]
    fn load_bytes_can_queue_distinct_snapshots_while_the_vm_is_not_consuming() {
        let budget = MovieBudgets::new(1, 1);
        let pending = budget.capture_snapshot(bytes::Bytes::from(vec![1; PAGE_BYTES]));
        assert_eq!(
            budget.input.permits.available_permits(),
            0,
            "admit at the AS call"
        );
        let second = budget.capture_snapshot(bytes::Bytes::from(vec![2; PAGE_BYTES]));
        let cancelled = budget.capture_snapshot(bytes::Bytes::from(vec![3; PAGE_BYTES]));
        drop(cancelled);
        let mut second = Box::pin(second);
        assert!(
            second.as_mut().now_or_never().is_none(),
            "spilled data must wait for input credit"
        );
        drop(pending);
        let input = futures::executor::block_on(second).unwrap();
        assert_eq!(input.bytes.as_ref(), vec![2; PAGE_BYTES]);
        drop(input);
        assert_eq!(budget.input.permits.available_permits(), 1);
    }

    #[test]
    fn local_file_fetch_waits_for_credit_before_the_navigator_buffers_its_body() {
        let budget = MovieBudgets::new(1, 1);
        let held = futures::executor::block_on(budget.snapshot(bytes::Bytes::from_static(b"held")));
        let started = Arc::new(AtomicBool::new(false));
        let flag = started.clone();
        let address = "file:///items.swf";
        let fetch = Box::pin(async move {
            flag.store(true, Ordering::SeqCst);
            Err(super::super::ErrorResponse {
                url: address.to_owned(),
                error: Error::FetchError("test read error".to_owned()),
            })
        });
        let mut next = Box::pin(budget.fetch(address, fetch));
        assert!(next.as_mut().now_or_never().is_none());
        assert!(!started.load(Ordering::SeqCst));
        drop(held);
        assert!(futures::executor::block_on(next).is_err());
        assert!(started.load(Ordering::SeqCst));
        assert_eq!(budget.input.permits.available_permits(), 1);
    }

    #[test]
    fn local_file_fetch_returns_unused_pages_when_the_actual_size_is_known() {
        use crate::backend::navigator::{FileNavigator, NavigatorBackend, Request};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.swf");
        std::fs::write(&path, b"small").unwrap();
        let url = url::Url::from_file_path(path).unwrap();
        let budget = MovieBudgets::new(3, 1);
        let fetch = FileNavigator::default().fetch_root(Request::get(url.to_string()));
        let (input, ..) = futures::executor::block_on(budget.fetch(url.as_str(), fetch))
            .map_err(|response| response.error)
            .unwrap();
        assert_eq!(input.bytes.as_ref(), b"small");
        assert_eq!(budget.input.permits.available_permits(), 2);
        drop(input);
        assert_eq!(budget.input.permits.available_permits(), 3);
    }

    #[test]
    fn http_fetch_can_receive_headers_while_movie_input_credit_is_exhausted() {
        let budget = MovieBudgets::new(1, 1);
        let _held = futures::executor::block_on(budget.snapshot(bytes::Bytes::new()));
        let address = "https://example.test/movie.swf";
        let fetch = Box::pin(async move {
            Err(super::super::ErrorResponse {
                url: address.to_owned(),
                error: Error::FetchError("test network error".to_owned()),
            })
        });
        assert!(matches!(
            budget.fetch(address, fetch).now_or_never(),
            Some(Err(_))
        ));
    }
}
