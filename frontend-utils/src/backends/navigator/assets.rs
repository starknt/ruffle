//! Session-local scheduling for explicitly buffered, public HTTP resources.
use super::fetch::{Response, ResponseBody};
use futures::future::{FutureExt, LocalBoxFuture, WeakShared};
use reqwest::{
    Client,
    cookie::{CookieStore, Jar},
    header,
};
use ruffle_core::backend::navigator::{
    ErrorResponse, FetchReason, OwnedFuture, SuccessResponse, get_encoding,
};
use ruffle_core::loader::Error;
use ruffle_core::swf::Encoding;
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::{Rc, Weak},
    sync::Arc,
    task::{Poll, Waker},
    time::{Duration, Instant},
};
use url::Url;

const MAX_ACTIVE: usize = 16;
const PER_ORIGIN: usize = 12;
const MAX_LOGICAL: usize = 256;
const MAX_BODY: usize = 4 * 1024 * 1024;
const MAX_SHARED: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy, Default, Debug)]
pub struct AssetStats {
    pub requested: u64,
    pub transfers: u64,
    pub joined: u64,
    pub cancelled: u64,
    pub peak_active: usize,
    pub peak_shared_bytes: usize,
    pub body_bytes: u64,
    pub delivered: u64,
    pub queue_ms: u64,
    pub headers_ms: u64,
    pub body_ms: u64,
}

type Transfer = LocalBoxFuture<'static, Rc<Outcome>>;

struct Waiting {
    id: u64,
    origin: String,
    visible: bool,
    since: Instant,
    waker: Waker,
}

#[derive(Default)]
struct State {
    groups: HashMap<Url, (u64, WeakShared<Transfer>)>,
    active: usize,
    origins: HashMap<String, usize>,
    waiting: Vec<Waiting>,
    shared_bytes: usize,
    seq: u64,
    stats: AssetStats,
}

#[derive(Clone)]
pub struct AssetScheduler {
    inner: Rc<RefCell<State>>,
    admissions: Arc<tokio::sync::Semaphore>,
    allow: fn(&Url) -> bool,
    enabled: Rc<Cell<bool>>,
    visible_assets: Rc<Cell<Option<fn(&Url) -> bool>>>,
}

impl AssetScheduler {
    pub(super) fn new(allow: fn(&Url) -> bool) -> Self {
        Self {
            inner: Rc::default(),
            admissions: Arc::new(tokio::sync::Semaphore::new(MAX_LOGICAL)),
            allow,
            enabled: Rc::new(Cell::new(true)),
            visible_assets: Rc::default(),
        }
    }
    pub fn enabled(&self) -> bool {
        self.enabled.get()
    }
    pub fn set_visible_assets(&self, filter: Option<fn(&Url) -> bool>) {
        self.visible_assets.set(filter);
    }
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.set(enabled);
    }
    pub fn stats(&self) -> AssetStats {
        self.inner.borrow().stats
    }
    pub(super) fn allows(&self, url: &Url) -> bool {
        self.enabled() && (self.allow)(url)
    }

    pub(super) fn fetch(
        &self,
        client: Client,
        jar: Arc<Jar>,
        url: Url,
        reason: FetchReason,
    ) -> OwnedFuture<Box<dyn SuccessResponse>, ErrorResponse> {
        let this = self.clone();
        Box::pin(async move {
            let _slot = this
                .admissions
                .acquire()
                .await
                .expect("session admission remains open");
            let visible = reason == FetchReason::LoadSwf
                || this.visible_assets.get().is_some_and(|f| f(&url));
            let (shared, leader) = {
                let mut state = this.inner.borrow_mut();
                state.stats.requested += 1;
                let existing = jar
                    .cookies(&url)
                    .is_none()
                    .then(|| state.groups.get(&url).and_then(|(_, g)| g.upgrade()))
                    .flatten();
                if let Some(existing) = existing {
                    state.stats.joined += 1;
                    (existing, false)
                } else {
                    state.seq += 1;
                    let id = state.seq;
                    let entry = Group {
                        state: Rc::downgrade(&this.inner),
                        url: url.clone(),
                        id,
                    };
                    let transfer =
                        this.download(client.clone(), jar.clone(), url.clone(), visible, true);
                    let future = async move {
                        let outcome = Rc::new(transfer.await);
                        drop(entry);
                        outcome
                    }
                    .boxed_local()
                    .shared();
                    state.groups.insert(
                        url.clone(),
                        (id, future.downgrade().expect("unpolled group")),
                    );
                    (future, true)
                }
            };
            let result = shared.await;
            if let Some(response) = result.for_reader(leader) {
                return response;
            }
            drop(result);
            // A private, varying, or oversized response belongs exclusively to its initiating reader.
            this.download(client, jar, url, visible, false)
                .await
                .for_reader(true)
                .expect("independent response")
        })
    }

    fn download(
        &self,
        client: Client,
        jar: Arc<Jar>,
        url: Url,
        visible: bool,
        may_share: bool,
    ) -> LocalBoxFuture<'static, Outcome> {
        let state = self.inner.clone();
        async move {
            let permit =
                TransferPermit::acquire(state.clone(), url.origin().ascii_serialization(), visible)
                    .await;
            let began = Instant::now();
            let anonymous = may_share && jar.cookies(&url).is_none();
            let send = client
                .get(url.clone())
                .header(header::CONTENT_TYPE, "")
                .send();
            let sent = run_io(send)
                .await
                .map_err(|e| (false, e.to_string()))
                .and_then(|r| r.map_err(|e| (e.is_connect(), e.to_string())));
            state.borrow_mut().stats.headers_ms += began.elapsed().as_millis() as u64;
            let mut http = match sent {
                Ok(response) => response,
                Err((connect, message)) => {
                    permit.complete();
                    return Outcome::Failure(Failure::Request(url.to_string(), connect, message));
                }
            };
            let meta = Meta::from_http(&http, &url);
            if !http.status().is_success() {
                permit.complete();
                return Outcome::HttpError(Failure::Http(
                    meta,
                    format!("Got {}", http.status()),
                    http.content_length().unwrap_or_default(),
                ));
            }
            let size = http
                .content_length()
                .and_then(|len| usize::try_from(len).ok());
            let budget = if anonymous
                && http.status() == reqwest::StatusCode::OK
                && meta.url == url.as_str()
                && public_headers(http.headers())
            {
                size.filter(|len| *len <= MAX_BODY)
                    .and_then(|len| Reservation::new(&state, len))
            } else {
                None
            };
            let mut prefix = Vec::new();
            let expected = http.content_length();
            if let Some(budget) = budget
                && prefix.try_reserve_exact(budget.bytes).is_ok()
            {
                let began = Instant::now();
                let size = budget.bytes;
                let read = run_io(read_body(http, prefix, size))
                    .await
                    .map_err(|e| e.to_string())
                    .and_then(|r| r.map_err(|e| e.to_string()));
                state.borrow_mut().stats.body_ms += began.elapsed().as_millis() as u64;
                match read {
                    Ok((data, rest)) => {
                        state.borrow_mut().stats.body_bytes += data.len() as u64;
                        if let Some(rest) = rest {
                            http = rest;
                            prefix = data;
                        } else {
                            permit.complete();
                            return Outcome::Shared(meta, Rc::new(SharedBody { data, budget }));
                        }
                    }
                    Err(message) => {
                        permit.complete();
                        return Outcome::BodyError(meta, message);
                    }
                }
            }
            Outcome::Single(RefCell::new(Some(Response {
                url: meta.url,
                text_encoding: meta.encoding,
                status: meta.status,
                redirected: meta.redirected,
                response_body: ResponseBody::Scheduled(Some(UnsharedBody {
                    http,
                    prefix,
                    expected,
                    permit,
                })),
            })))
        }
        .boxed_local()
    }
}

fn public_headers(headers: &header::HeaderMap) -> bool {
    !headers.contains_key(header::SET_COOKIE)
        && !headers.contains_key(header::CONTENT_RANGE)
        && [header::CACHE_CONTROL, header::VARY].iter().all(|name| {
            headers.get_all(name).iter().all(|v| {
                v.to_str().is_ok_and(|value| {
                    value.split(',').all(|part| {
                        let token = part.trim().split('=').next().unwrap_or("").trim();
                        if name == header::VARY {
                            token.eq_ignore_ascii_case("accept-encoding")
                        } else {
                            !["private", "no-store", "no-cache"]
                                .iter()
                                .any(|key| token.eq_ignore_ascii_case(key))
                        }
                    })
                })
            })
        })
}

struct Group {
    state: Weak<RefCell<State>>,
    url: Url,
    id: u64,
}
impl Drop for Group {
    fn drop(&mut self) {
        if let Some(state) = self.state.upgrade() {
            let mut state = state.borrow_mut();
            if state
                .groups
                .get(&self.url)
                .is_some_and(|(id, _)| *id == self.id)
            {
                state.groups.remove(&self.url);
            }
        }
    }
}

struct Queued {
    state: Rc<RefCell<State>>,
    id: u64,
}
impl Drop for Queued {
    fn drop(&mut self) {
        let mut s = self.state.borrow_mut();
        if let Some(index) = s.waiting.iter().position(|w| w.id == self.id) {
            s.waiting.remove(index);
            s.stats.cancelled += 1;
            for w in &s.waiting {
                w.waker.wake_by_ref()
            }
        }
    }
}

struct TransferPermit {
    state: Rc<RefCell<State>>,
    origin: String,
    complete: Cell<bool>,
}
impl TransferPermit {
    async fn acquire(state: Rc<RefCell<State>>, origin: String, visible: bool) -> Self {
        let id = {
            let mut s = state.borrow_mut();
            s.seq += 1;
            s.seq
        };
        let queued = Queued {
            state: state.clone(),
            id,
        };
        let since = Instant::now();
        std::future::poll_fn(|cx| {
            let mut s = state.borrow_mut();
            if let Some(w) = s.waiting.iter_mut().find(|w| w.id == id) {
                w.waker.clone_from(cx.waker());
            } else {
                s.waiting.push(Waiting {
                    id,
                    origin: origin.clone(),
                    visible,
                    since,
                    waker: cx.waker().clone(),
                });
            }
            if s.active < MAX_ACTIVE {
                let next = s
                    .waiting
                    .iter()
                    .filter(|w| s.origins.get(&w.origin).copied().unwrap_or(0) < PER_ORIGIN)
                    .min_by_key(|w| {
                        (
                            !(w.visible || w.since.elapsed() >= Duration::from_secs(1)),
                            w.id,
                        )
                    });
                if next.is_some_and(|w| w.id == id) {
                    s.waiting.retain(|w| w.id != id);
                    s.active += 1;
                    *s.origins.entry(origin.clone()).or_default() += 1;
                    s.stats.transfers += 1;
                    s.stats.peak_active = s.stats.peak_active.max(s.active);
                    s.stats.queue_ms += since.elapsed().as_millis() as u64;
                    for w in &s.waiting {
                        w.waker.wake_by_ref();
                    }
                    return Poll::Ready(());
                }
            }
            Poll::Pending
        })
        .await;
        drop(queued);
        Self {
            state,
            origin,
            complete: Cell::new(false),
        }
    }
    fn complete(&self) {
        self.complete.set(true);
    }
}
impl Drop for TransferPermit {
    fn drop(&mut self) {
        let mut s = self.state.borrow_mut();
        s.active -= 1;
        let count = s.origins.get_mut(&self.origin).expect("held origin");
        *count -= 1;
        if *count == 0 {
            s.origins.remove(&self.origin);
        }
        if !self.complete.get() {
            s.stats.cancelled += 1;
        }
        for w in &s.waiting {
            w.waker.wake_by_ref();
        }
    }
}

struct Reservation {
    state: Rc<RefCell<State>>,
    bytes: usize,
}
impl Reservation {
    fn new(state: &Rc<RefCell<State>>, bytes: usize) -> Option<Self> {
        let mut s = state.borrow_mut();
        if bytes > MAX_SHARED - s.shared_bytes {
            return None;
        }
        s.shared_bytes += bytes;
        s.stats.peak_shared_bytes = s.stats.peak_shared_bytes.max(s.shared_bytes);
        Some(Self {
            state: state.clone(),
            bytes,
        })
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.state.borrow_mut().shared_bytes -= self.bytes;
    }
}

pub(super) struct SharedBody {
    data: Vec<u8>,
    budget: Reservation,
}
impl SharedBody {
    pub fn bytes(&self) -> Vec<u8> {
        self.budget.state.borrow_mut().stats.delivered += 1;
        self.data.clone()
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
}
pub(super) struct UnsharedBody {
    http: reqwest::Response,
    prefix: Vec<u8>,
    pub expected: Option<u64>,
    permit: TransferPermit,
}
impl UnsharedBody {
    pub async fn read(self) -> Result<Vec<u8>, Error> {
        let began = Instant::now();
        let prefix_bytes = self.prefix.len();
        let result = run_io(read_body(self.http, self.prefix, usize::MAX)).await;
        self.permit.complete();
        let (data, _) = result
            .map_err(|e| Error::FetchError(e.to_string()))?
            .map_err(|e| Error::FetchError(e.to_string()))?;
        let mut state = self.permit.state.borrow_mut();
        state.stats.delivered += 1;
        state.stats.body_bytes += (data.len() - prefix_bytes) as u64;
        state.stats.body_ms += began.elapsed().as_millis() as u64;
        Ok(data)
    }
}

// The game thread can spend an entire frame decoding ABC. HTTP should keep making progress
// during that frame, and dropping the last waiter must cancel the owned I/O task.
async fn run_io<F: std::future::Future + Send + 'static>(
    future: F,
) -> Result<F::Output, tokio::task::JoinError>
where
    F::Output: Send + 'static,
{
    struct AbortOnDrop(tokio::task::AbortHandle);
    impl Drop for AbortOnDrop {
        fn drop(&mut self) {
            self.0.abort();
        }
    }
    let task = tokio::spawn(future);
    let _cancel = AbortOnDrop(task.abort_handle());
    tokio::task::unconstrained(task).await
}

async fn read_body(
    mut response: reqwest::Response,
    mut data: Vec<u8>,
    limit: usize,
) -> Result<(Vec<u8>, Option<reqwest::Response>), reqwest::Error> {
    while let Some(chunk) = response.chunk().await? {
        data.extend_from_slice(&chunk);
        if data.len() > limit {
            return Ok((data, Some(response)));
        }
    }
    Ok((data, None))
}

struct Meta {
    url: String,
    status: u16,
    redirected: bool,
    encoding: Option<&'static Encoding>,
}
impl Meta {
    fn from_http(http: &reqwest::Response, original: &Url) -> Self {
        Self {
            url: http.url().to_string(),
            status: http.status().as_u16(),
            redirected: http.url() != original,
            encoding: http
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .and_then(get_encoding),
        }
    }
    fn response(&self, response_body: ResponseBody) -> Box<dyn SuccessResponse> {
        Box::new(Response {
            url: self.url.clone(),
            status: self.status,
            redirected: self.redirected,
            text_encoding: self.encoding,
            response_body,
        })
    }
}

enum Failure {
    Request(String, bool, String),
    Http(Meta, String, u64),
}
impl Failure {
    fn response(&self) -> ErrorResponse {
        match self {
            Self::Request(url, connect, reason) => ErrorResponse {
                url: url.clone(),
                error: if *connect {
                    Error::InvalidDomain(url.clone())
                } else {
                    Error::FetchError(reason.clone())
                },
            },
            Self::Http(m, label, len) => ErrorResponse {
                url: m.url.clone(),
                error: Error::HttpNotOk(label.clone(), m.status, m.redirected, *len),
            },
        }
    }
}
enum Outcome {
    Shared(Meta, Rc<SharedBody>),
    Single(RefCell<Option<Response>>),
    Failure(Failure),
    HttpError(Failure),
    BodyError(Meta, String),
}
impl Outcome {
    fn for_reader(&self, leader: bool) -> Option<Result<Box<dyn SuccessResponse>, ErrorResponse>> {
        match self {
            Self::Shared(meta, body) => {
                Some(Ok(meta.response(ResponseBody::Shared(Some(body.clone())))))
            }
            Self::BodyError(meta, message) => Some(Ok(meta.response(ResponseBody::File(Err(
                std::io::Error::other(message.clone()),
            ))))),
            Self::HttpError(f) if leader => Some(Err(f.response())),
            Self::HttpError(_) => None,
            Self::Failure(f) => Some(Err(f.response())),
            Self::Single(r) if leader => Some(Ok(Box::new(
                r.borrow_mut().take().expect("original reader"),
            ))),
            Self::Single(_) => None,
        }
    }
}
