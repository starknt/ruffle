//! Preparation owns transport bytes. Only the finished movie or non-SWF body crosses back to the player.
mod payload;
mod snapshot;
use super::{ContentType, Error, Instant, LoadBytesInfo, SwfMovieData};
use crate::backend::navigator::LoaderRequest;
use payload::MoviePayload;

pub(super) enum PreparedContent {
    Movie(SwfMovieData),
    Image {
        source: bytes::Bytes,
        pixels: Result<ImagePixels, String>,
    },
    Other(bytes::Bytes),
}

pub(super) struct ImagePixels {
    pub width: u32,
    pub height: u32,
    pub transparency: bool,
    pub loader_request: Option<LoaderRequest>,
    pub colors: Vec<crate::bitmap::bitmap_data::Color>,
}

impl ImagePixels {
    pub(super) fn decode(
        source: &[u8],
        loader_request: Option<LoaderRequest>,
    ) -> Result<Self, String> {
        let bitmap = ruffle_render::utils::decode_define_bits_jpeg(source, None)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            loader_request,
            width: bitmap.width(),
            height: bitmap.height(),
            transparency: bitmap.format().supports_transparency(),
            colors: bitmap
                .as_colors()
                .map(crate::bitmap::bitmap_data::Color::from)
                .collect(),
        })
    }
}

impl PreparedContent {
    pub(super) fn into_movie(self) -> Result<SwfMovieData, Error> {
        match self {
            Self::Movie(movie) => Ok(movie),
            Self::Image { source: bytes, .. } | Self::Other(bytes) => Err(Error::UnexpectedData(
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

// Semaphore units control admission. The charge travels with the payload and
// can exceed the shared window when one subject owns every unit.
pub(super) struct ByteCharge {
    permits: OwnedSemaphorePermit,
    pub bytes: usize,
}

pub(super) struct Delivery {
    pub pending: ByteCharge,
    pub decode_peak: usize,
}

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

    fn units(&self, bytes: Option<usize>) -> u32 {
        bytes.map_or(self.pages, |n| {
            n.div_ceil(PAGE_BYTES).clamp(1, self.pages as usize) as u32
        })
    }

    async fn reserve(&self, bytes: Option<usize>) -> ByteCharge {
        let permits = self
            .permits
            .clone()
            .acquire_many_owned(self.units(bytes))
            .await
            .expect("permanent movie window");
        ByteCharge {
            permits,
            bytes: bytes.unwrap_or(0),
        }
    }

    fn settle(&self, mut charge: ByteCharge, bytes: usize) -> ByteCharge {
        let permits = charge
            .permits
            .split(self.units(Some(bytes)) as usize)
            .expect("admitted movie payload");
        ByteCharge { permits, bytes }
    }
}

#[derive(Clone)]
pub(super) struct MovieBudgets {
    input: ByteWindow,
    decoded: ByteWindow,
    pending: ByteWindow,
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
    pub loader_request: Option<LoaderRequest>,
    pub abort: Option<futures::future::AbortHandle>,
    reservation: ByteCharge,
}

impl MovieBudgets {
    fn new(input_pages: u32, decoded_pages: u32) -> Self {
        Self {
            input: ByteWindow::new(input_pages),
            decoded: ByteWindow::new(decoded_pages),
            pending: ByteWindow::new(decoded_pages),
            spill: Default::default(),
            transfers: Arc::new(Semaphore::new(MAX_PREPARATIONS)),
        }
    }

    #[cfg(test)]
    async fn snapshot(&self, bytes: bytes::Bytes) -> MovieInput {
        let reservation = self.input.reserve(Some(bytes.len())).await;
        MovieInput {
            bytes,
            loader_request: None,
            abort: None,
            reservation,
        }
    }

    pub(super) async fn reserve_before_fetch(&self, address: &str) -> Option<ByteCharge> {
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
        reservation: Option<ByteCharge>,
        loader_request: Option<LoaderRequest>,
    ) -> Result<MovieInput, Error> {
        if let Some(echo) = response.loader_request()
            && Some(echo) != loader_request
        {
            return Err(Error::FetchError("Mismatched Loader response".into()));
        }
        let buffered = response.buffered_body_len();
        let reservation = match reservation {
            Some(permit) => permit,
            None => self.input.reserve(buffered).await,
        };
        // Unknown-length bodies already own the whole window before polling
        // the producer. Never upgrade a partial reservation while reading.
        let bytes = response.body_bytes().await?;
        if buffered.is_some_and(|bound| bytes.len() > bound) {
            return Err(Error::FetchError(
                "Movie body exceeded its buffered length".into(),
            ));
        }
        let held = self.input.settle(reservation, bytes.len());
        Ok(MovieInput {
            bytes,
            loader_request,
            abort: None,
            reservation: held,
        })
    }

    pub(super) async fn fetch(
        &self,
        address: &str,
        fetch: super::OwnedFuture<Box<dyn super::SuccessResponse>, super::ErrorResponse>,
        loader_request: Option<LoaderRequest>,
    ) -> Result<(MovieInput, String, u16, bool), super::ErrorResponse> {
        let reservation = self.reserve_before_fetch(address).await;
        let response = fetch.await?;
        let url = response.url().into_owned();
        let status = response.status();
        let redirected = response.redirected();
        let input = self
            .read(response, reservation, loader_request)
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
    ) -> (Result<(PreparedContent, Delivery), swf::error::Error>, u64) {
        let start = Instant::now();
        let kind = ContentType::sniff(&input.bytes);
        let needs_probe = MoviePayload::needs_exclusive_probe(&input.bytes);
        let sized = if needs_probe {
            None
        } else {
            match MoviePayload::inspect(input.bytes.clone()) {
                Ok(payload) => Some(payload),
                Err(error) => return (Err(error), start.elapsed().as_nanos() as u64),
            }
        };
        let window = budget.decoded.clone();
        let decoded = window
            .reserve(sized.as_ref().map(|payload| payload.decode_peak))
            .await;
        let inline = matches!(self, Self::Inline)
            && (use_inline_movie_parse(&url) || (!needs_probe && !kind.is_image()));
        let current = move || {
            if input.abort.as_ref().is_some_and(|read| read.is_aborted()) {
                Err(swf::error::Error::from(std::io::Error::from(
                    std::io::ErrorKind::Interrupted,
                )))
            } else {
                Ok(())
            }
        };
        let work = move || {
            current()?;
            let payload = match sized {
                Some(payload) => payload,
                None => MoviePayload::inspect(input.bytes.clone())?,
            };
            let decoded = window.settle(decoded, payload.decode_peak);
            let content = prepare_payload(&payload, url, loader_url, info, input.loader_request)?;
            current()?;
            let retained = payload.handoff_bytes(&content);
            if let Some(tag) = input.loader_request
                && matches!(content, PreparedContent::Image { pixels: Ok(_), .. })
            {
                tracing::debug!(target: "ruffle_loader", stage = "decode",
                    session_generation = tag.session_generation, request_id = tag.request_id,
                    resource_version = tag.resource_version, input_bytes = payload.transport_len,
                    input_credit_bytes = input.reservation.bytes, decoded_credit_bytes = decoded.bytes,
                    handoff_bytes = retained);
            }
            drop(input.bytes);
            drop(input.reservation);
            Ok((content, retained, decoded))
        };
        let prepared = if inline {
            work()
        } else {
            // Never reacquire a transfer slot while holding input credit.
            let worker = PREPARATIONS
                .clone()
                .acquire_owned()
                .await
                .expect("permanent movie pool");
            run_worker(worker, move || {
                let _transfer = match self {
                    Self::Worker(permit) => Some(permit),
                    Self::Inline => None,
                };
                work()
            })
            .await
        };
        let elapsed = start.elapsed().as_nanos() as u64;
        let result = match prepared {
            Ok((content, retained, decoded)) => {
                // No decoder thread or player lock is held while a slow VM
                // applies backpressure. Cancellation drops content and credit together.
                let pending = budget.pending.reserve(Some(retained)).await;
                let delivery = Delivery {
                    pending,
                    decode_peak: decoded.bytes,
                };
                drop(decoded);
                Ok((content, delivery))
            }
            Err(error) => Err(error),
        };
        (result, elapsed)
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
        .map(|(content, _)| content)
}

#[cfg(test)]
pub(super) fn prepare_swf_movie_data(
    body: impl Into<bytes::Bytes>,
    url: String,
    loader_url: Option<String>,
    load_bytes_info: Option<LoadBytesInfo>,
    loader_request: Option<LoaderRequest>,
) -> Result<PreparedContent, swf::error::Error> {
    let payload = MoviePayload::inspect(body.into())?;
    prepare_payload(&payload, url, loader_url, load_bytes_info, loader_request)
}

fn prepare_payload(
    payload: &MoviePayload,
    url: String,
    loader_url: Option<String>,
    load_bytes_info: Option<LoadBytesInfo>,
    loader_request: Option<LoaderRequest>,
) -> Result<PreparedContent, swf::error::Error> {
    let source = &payload.source;
    if payload.kind.is_image() {
        let pixels = match &payload.image_error {
            Some(error) => Err(error.clone()),
            None => ImagePixels::decode(source, loader_request),
        };
        return Ok(PreparedContent::Image {
            source: source.clone(),
            pixels,
        });
    }
    if payload.kind != ContentType::Swf {
        return Ok(PreparedContent::Other(source.clone()));
    }
    let movie = SwfMovieData::from_data(source, url, loader_url, load_bytes_info)?
        .with_shared_bytes(true)
        .with_compressed_len(payload.transport_len);
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
    fn two_pending_handoffs_backpressure_decode_and_cancel_refunds_each_owner() {
        let budget = MovieBudgets::new(1, 2);
        let bytes = bytes::Bytes::from(vec![7; PAGE_BYTES]);
        futures::executor::block_on(async {
            let load = || async {
                let input = budget.capture_snapshot(bytes.clone()).await.unwrap();
                Preparation::Inline
                    .prepare(input, &budget, String::new(), None, None)
                    .await
                    .0
                    .unwrap()
            };
            let first = load().await;
            let second = load().await;
            assert_eq!(
                first.1.pending.bytes + second.1.pending.bytes,
                2 * PAGE_BYTES
            );
            assert_eq!(budget.pending.permits.available_permits(), 0);
            assert_eq!(budget.decoded.permits.available_permits(), 2);
            let mut blocked = Box::pin(load());
            assert!(blocked.as_mut().now_or_never().is_none());
            assert_eq!(budget.input.permits.available_permits(), 1);
            assert_eq!(budget.decoded.permits.available_permits(), 1);
            drop(blocked);
            assert_eq!(budget.decoded.permits.available_permits(), 2);
            let mut next = Box::pin(load());
            assert!(next.as_mut().now_or_never().is_none());
            drop(second);
            let resumed = next.await;
            drop((first, resumed));
            assert_eq!(budget.pending.permits.available_permits(), 2);
        });
    }

    #[test]
    fn session_teardown_drops_pending_deliveries_and_a_blocked_root_movie() {
        use crate::player::PlayerBuilder;
        let budget = MovieBudgets::new(1, 2);
        let mut session = PlayerBuilder::new().build_session();
        session.lock().unwrap().mutate_with_update_context(|uc| {
            uc.load_manager.movie_budgets = budget.clone();
            let slow_vm = budget.clone();
            uc.spawn_future(Box::pin(async move {
                let mut deliveries = Vec::new();
                for _ in 0..2 {
                    let input = slow_vm
                        .capture_snapshot(bytes::Bytes::from_static(b"pending"))
                        .await?;
                    deliveries.push(
                        Preparation::Inline
                            .prepare(input, &slow_vm, String::new(), None, None)
                            .await
                            .0?,
                    );
                }
                futures::future::pending::<()>().await;
                drop(deliveries);
                Ok(())
            }));
        });
        session.poll_tasks();
        assert_eq!(budget.pending.permits.available_permits(), 0);
        let swf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/tests/swfs/avm1/asnew/test.swf");
        session.lock().unwrap().fetch_root_movie(
            url::Url::from_file_path(swf).unwrap().into(),
            vec![],
            Box::new(|_| panic!("root committed after teardown")),
        );
        session.poll_tasks();
        assert_eq!(budget.input.permits.available_permits(), 1);
        assert_eq!(budget.decoded.permits.available_permits(), 1);
        drop(session);
        assert_eq!(budget.decoded.permits.available_permits(), 2);
        assert_eq!(budget.pending.permits.available_permits(), 2);
    }

    #[test]
    fn retiring_a_child_movie_wakes_the_host_executor_and_refunds_its_bytes() {
        use crate::{backend::tasks::LocalExecutor, player::PlayerBuilder};

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/tests/swfs/avm2/loader_image/test.swf")
            .canonicalize()
            .unwrap();
        let movie = SwfMovieData::from_data(
            &std::fs::read(&path).unwrap(),
            url::Url::from_file_path(path).unwrap().into(),
            None,
            None,
        )
        .unwrap();
        for replace_root in [false, true] {
            for before_fetch in [false, true] {
                let budget = MovieBudgets::new(1, 1);
                let window = if before_fetch {
                    &budget.input
                } else {
                    &budget.pending
                };
                let held = futures::executor::block_on(window.reserve(None));
                let mut executor = LocalExecutor::new();
                let renderer = crate::bitmap::symbol_pixels::tests::renderer(false);
                let uploads = renderer.loader_uploads.clone();
                let player = PlayerBuilder::new()
                    .with_movie(movie.clone())
                    .with_renderer(renderer)
                    .build_with_spawner(executor.spawner());
                player.lock().unwrap().mutate_with_update_context(|uc| {
                    uc.load_manager.movie_budgets = budget.clone()
                });
                player.lock().unwrap().run_frame();
                executor.run();
                assert_eq!(executor.pending_tasks(), 1);
                assert_eq!(
                    budget.decoded.permits.available_permits(),
                    usize::from(before_fetch)
                );
                if replace_root {
                    player.lock().unwrap().mutate_with_update_context(|uc| {
                        uc.replace_root_movie(SwfMovieData::empty(8, None));
                        assert_eq!(uc.load_manager.pending_counts(), (0, 0));
                    });
                    executor.run();
                    player.lock().unwrap().render().unwrap();
                }
                let weak = Arc::downgrade(&player);
                drop(player);
                assert!(
                    weak.upgrade().is_none(),
                    "child read kept the session alive"
                );
                executor.run();
                assert_eq!(executor.pending_tasks(), 0, "stale read waited for credit");
                assert!(uploads.borrow().is_empty());
                drop(held);
                for window in [&budget.input, &budget.decoded, &budget.pending] {
                    assert_eq!(window.permits.available_permits(), 1);
                }
            }
        }
    }

    #[test]
    fn parse_and_spill_errors_release_credit_while_vm_delivery_is_full() {
        let mut budget = MovieBudgets::new(1, 1);
        let dir = tempfile::tempdir().unwrap();
        budget.set_snapshot_directory(dir.path().join("missing-directory"));
        futures::executor::block_on(async {
            let pending = budget.pending.reserve(None).await;
            // Normal header and declared length, truncated before the frame rectangle.
            let bytes = bytes::Bytes::from_static(b"FWS\x09\x20\0\0\0");
            let input = budget.capture_snapshot(bytes.clone()).await.unwrap();
            assert!(budget.capture_snapshot(bytes).await.is_err());
            let (parsed, _) = Preparation::Inline
                .prepare(input, &budget, String::new(), None, None)
                .now_or_never()
                .expect("parse errors bypass pending delivery");
            assert!(parsed.is_err());
            assert_eq!(budget.input.permits.available_permits(), 1);
            assert_eq!(budget.decoded.permits.available_permits(), 1);
            drop(pending);
            assert_eq!(budget.pending.permits.available_permits(), 1);
        });
    }

    #[test]
    fn compressed_movie_handoff_preserves_mapped_bytes_and_snapshots_on_write() {
        use crate::{avm2::bytearray::ByteArrayStorage, player::PlayerBuilder};
        let data = vec![0x5a; PAGE_BYTES + 123];
        let mut cws = Vec::new();
        let mut header = swf::Header::default_with_swf_version(10);
        header.compression = swf::Compression::Zlib;
        swf::write_swf(
            &header,
            &[swf::Tag::DefineBinaryData(swf::DefineBinaryData {
                id: 1,
                data: &data,
            })],
            &mut cws,
        )
        .unwrap();
        assert!(cws.len() < PAGE_BYTES);
        let budget = MovieBudgets::new(1, 1);
        let input = futures::executor::block_on(budget.capture_snapshot(cws.into())).unwrap();
        let (prepared, _) = futures::executor::block_on(Preparation::Inline.prepare(
            input,
            &budget,
            "file:///mapped.swf".into(),
            None,
            None,
        ));
        let (movie, delivery) = prepared.unwrap();
        let file = movie.into_movie().unwrap().shared_file().unwrap();
        assert!(file.private_pages().is_some());
        assert!(delivery.pending.bytes >= file.file().len());
        assert!(file.file().len() > PAGE_BYTES);
        let player = PlayerBuilder::new().build();
        player.lock().unwrap().mutate_with_update_context(|uc| {
            let mut array = ByteArrayStorage::new(uc);
            array.set_shared_movie(file.clone());
            let captured = budget.capture_snapshot(array.snapshot());
            array.write_at(&[0x77], 1024).unwrap();
            let input = futures::executor::block_on(captured).unwrap();
            assert_eq!(file.file().as_ptr(), input.bytes.as_ptr());
            assert_eq!(input.bytes[1024], 0x5a);
            assert_eq!(array.bytes()[1024], 0x77);
            assert_eq!(input.reservation.bytes, file.file().len());
        });
        drop(delivery);
        assert_eq!(budget.pending.permits.available_permits(), 1);
    }

    #[test]
    fn image_preparation_reserves_decoded_pixels_before_decoding_outside_the_vm() {
        let png = bytes::Bytes::from_static(include_bytes!(
            "../../../tests/tests/swfs/avm2/loader_load/test.png"
        ));
        let budget = MovieBudgets::new(2, 4);
        futures::executor::block_on(async {
            let input = budget.snapshot(png.clone()).await;
            let (result, _) = Preparation::Inline
                .prepare(input, &budget, "file:///image.png".into(), None, None)
                .await;
            let (content, delivery) = result.unwrap();
            assert_eq!(
                delivery.pending.permits.num_permits(),
                4,
                "exclusive 600 by 400 RGBA delivery"
            );
            assert_eq!(delivery.decode_peak, 1_920_000 + png.len());
            assert_eq!(delivery.pending.bytes, 960_000 + png.len());
            let PreparedContent::Image {
                source,
                pixels: Ok(image),
            } = content
            else {
                panic!("expected prepared pixels and original loader bytes")
            };
            assert_eq!(source, png);
            assert_eq!(
                (image.width, image.height, image.colors.len()),
                (600, 400, 240000)
            );
            assert!(budget.decoded.reserve(None).now_or_never().is_some());
            assert!(budget.pending.reserve(Some(1)).now_or_never().is_none());
            drop((image, source, delivery));
            assert!(budget.pending.reserve(None).now_or_never().is_some());
        });
    }

    #[test]
    fn admitted_image_finishes_while_other_transfers_wait_for_input() {
        let budget = MovieBudgets::new(1, 1);
        let transfers = budget
            .transfers
            .clone()
            .try_acquire_many_owned(MAX_PREPARATIONS as u32)
            .unwrap();
        let png = bytes::Bytes::from_static(include_bytes!(
            "../../../tests/tests/swfs/avm2/loader_load/test.png"
        ));
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let ok = futures::executor::block_on(async {
                let input = budget.snapshot(png).await;
                let (prepared, _) = Preparation::for_bytes(&budget)
                    .prepare(input, &budget, String::new(), None, None)
                    .await;
                matches!(
                    prepared,
                    Ok((PreparedContent::Image { pixels: Ok(_), .. }, _))
                )
            });
            tx.send(ok).unwrap();
        });
        let finished = rx.recv_timeout(std::time::Duration::from_secs(3));
        drop(transfers);
        worker.join().unwrap();
        assert_eq!(finished, Ok(true));
    }

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
            let input = budget.capture_snapshot(bytes.clone()).await.unwrap();
            assert_eq!(input.reservation.bytes, 3 * PAGE_BYTES);
            assert!(budget.snapshot(bytes.clone()).now_or_never().is_none());
            let (result, _) = Preparation::Inline
                .prepare(input, &budget, String::new(), None, None)
                .await;
            assert!(
                matches!(result, Ok((PreparedContent::Other(_), ref delivery)) if delivery.pending.bytes == bytes.len() && delivery.decode_peak == bytes.len())
            );
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
            drop(result);
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
        let mut next = Box::pin(budget.fetch(address, fetch, None));
        assert!(next.as_mut().now_or_never().is_none());
        assert!(!started.load(Ordering::SeqCst));
        drop(held);
        assert!(futures::executor::block_on(next).is_err());
        assert!(started.load(Ordering::SeqCst));
        assert_eq!(budget.input.permits.available_permits(), 1);
    }

    #[test]
    fn response_for_another_loader_version_releases_its_input_credit() {
        use crate::backend::navigator::{FileNavigator, NavigatorBackend, Request};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.png");
        std::fs::write(&path, b"unexpected body").unwrap();
        let url = url::Url::from_file_path(&path).unwrap().to_string();
        let mut navigator = FileNavigator::default();
        navigator.set_root_url(&url);
        let budget = MovieBudgets::new(1, 1);
        let current = LoaderRequest {
            session_generation: 8,
            request_id: 11,
            resource_version: 2,
        };
        let mut request = Request::get(url.clone());
        request.set_loader_request(LoaderRequest {
            resource_version: 1,
            ..current
        });
        let response = futures::executor::block_on(budget.fetch(
            &url,
            navigator.fetch(request),
            Some(current),
        ));
        assert!(
            matches!(response, Err(super::super::ErrorResponse { error: Error::FetchError(message), .. }) if message == "Mismatched Loader response")
        );
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
        let (input, ..) = futures::executor::block_on(budget.fetch(url.as_str(), fetch, None))
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
            budget.fetch(address, fetch, None).now_or_never(),
            Some(Err(_))
        ));
    }
}
