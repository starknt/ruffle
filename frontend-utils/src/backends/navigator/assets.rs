//! Session-local scheduling for explicitly buffered, public HTTP resources.
#[cfg(test)]
#[path = "asset_body_tests.rs"]
mod tests;
use super::fetch::{Response, ResponseBody};
use futures::channel::oneshot;
use futures::future::{FutureExt, LocalBoxFuture, WeakShared};
use reqwest::{Client, cookie::CookieStore, header};
use ruffle_core::backend::navigator::{
    Bytes, ErrorResponse, FetchReason, OwnedFuture, SuccessResponse, get_encoding,
};
use ruffle_core::loader::Error;
use ruffle_core::memory::{MemoryBuffer, MemoryKind, MemoryScope, PressureLevel};
use ruffle_core::swf::Encoding;
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::{Rc, Weak},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Poll, Waker},
    time::{Duration, Instant},
};
use url::Url;

pub struct CachedAsset {
    pub bytes: Vec<u8>,
    pub encoding: Option<String>,
    pub etag: Option<String>,
    pub fresh_until: i64,
}
/// Storage for versioned static assets. All calls run on a blocking I/O worker.
pub trait AssetVfs: Send + Sync {
    fn get(&self, key: &str) -> Option<CachedAsset>;
    fn validator(&self, key: &str) -> Option<String>;
    fn put(
        &self,
        key: &str,
        encoding: Option<&str>,
        etag: Option<&str>,
        fresh_until: i64,
        body: &[u8],
    ) -> bool;
}

const MAX_ACTIVE: usize = 16;
const PER_ORIGIN: usize = 12;
const MAX_LOGICAL: usize = 256;
const MAX_BODY: usize = 4 * 1024 * 1024;
const MAX_SHARED: usize = 32 * 1024 * 1024;
const MAX_PRELOADS: usize = 24;
const MAX_ACTIVE_PRELOADS: usize = 4;
const PRELOAD_LIFETIME: Duration = Duration::from_secs(20);

struct Preload {
    cookies: (bool, u64),
    started: Instant,
    claimed: Rc<Cell<bool>>,
    ready: oneshot::Receiver<Option<Response>>,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct AssetStats {
    pub requested: u64,
    pub seen: u64,
    pub policy_bypass: u64,
    pub credential_bypass: u64,
    pub revalidated: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_writes: u64,
    pub cache_bytes: u64,
    pub transfers: u64,
    pub joined: u64,
    pub preloaded: u64,
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
    shared_bytes: Arc<AtomicUsize>,
    seq: u64,
    stats: AssetStats,
}

#[derive(Clone)]
pub struct AssetScheduler {
    inner: Rc<RefCell<State>>,
    memory: Rc<RefCell<MemoryScope>>,
    preloads: Rc<RefCell<HashMap<Url, Preload>>>,
    preload_slots: Arc<tokio::sync::Semaphore>,
    admissions: Arc<tokio::sync::Semaphore>,
    disk_reads: Arc<tokio::sync::Semaphore>,
    allow: fn(&Url) -> bool,
    vfs: Rc<RefCell<Option<Arc<dyn AssetVfs>>>>,
    account_vfs: Rc<RefCell<Option<Arc<dyn AssetVfs>>>>,
    enabled: Rc<Cell<bool>>,
    cache_first: Rc<Cell<bool>>,
    visible_assets: Rc<Cell<Option<fn(&Url) -> bool>>>,
}

impl AssetScheduler {
    pub(super) fn new(allow: fn(&Url) -> bool, memory: Rc<RefCell<MemoryScope>>) -> Self {
        Self {
            inner: Rc::default(),
            memory,
            preloads: Rc::default(),
            preload_slots: Arc::new(tokio::sync::Semaphore::new(MAX_ACTIVE_PRELOADS)),
            admissions: Arc::new(tokio::sync::Semaphore::new(MAX_LOGICAL)),
            disk_reads: Arc::new(tokio::sync::Semaphore::new(1)),
            allow,
            vfs: Rc::default(),
            account_vfs: Rc::default(),
            enabled: Rc::new(Cell::new(true)),
            cache_first: Rc::new(Cell::new(false)),
            visible_assets: Rc::default(),
        }
    }
    pub fn mount_vfs(&self, public: Arc<dyn AssetVfs>, account: Arc<dyn AssetVfs>) {
        *self.vfs.borrow_mut() = Some(public);
        *self.account_vfs.borrow_mut() = Some(account);
    }
    pub fn cache_enabled(&self) -> bool {
        self.vfs.borrow().is_some()
    }
    pub(super) fn observe(&self, eligible: bool, cookie: bool) {
        let mut s = self.inner.borrow_mut();
        s.stats.seen += 1;
        if !eligible {
            s.stats.policy_bypass += 1;
        } else if cookie {
            s.stats.credential_bypass += 1;
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
        if !enabled {
            self.clear_preloads();
        }
    }
    /// Prefer an existing disk entry even after its HTTP freshness window.
    /// Public game URLs are versioned by the Aola allow-list, so this avoids
    /// re-fetching an already cached SWF or static asset during offline play.
    pub fn set_cache_first(&self, enabled: bool) {
        self.cache_first.set(enabled);
    }
    pub fn stats(&self) -> AssetStats {
        self.inner.borrow().stats
    }
    /// Release any unclaimed hints when the loading phase ends.
    pub fn clear_preloads(&self) {
        self.preloads.borrow_mut().clear();
    }
    pub(super) fn allows(&self, url: &Url) -> bool {
        (self.enabled() || self.cache_enabled()) && (self.allow)(url)
    }

    /// Prepare one ordinary buffered GET. Only cache-admissible responses can
    /// be handed to a loader, once, in the identical session cookie context.
    pub(super) fn preload(
        &self,
        client: Client,
        jar: Arc<super::SessionCookies>,
        url: Url,
    ) -> Option<OwnedFuture<(), Error>> {
        if !self.cache_enabled() || !self.enabled() || !self.allows(&url) {
            return None;
        }
        let weight = match self.memory.borrow().manager().pressure().level {
            PressureLevel::Normal => 1,
            PressureLevel::Moderate => 2,
            PressureLevel::High | PressureLevel::Critical => {
                self.clear_preloads();
                return None;
            }
        };
        let mut tickets = self.preloads.borrow_mut();
        tickets.retain(|_, ticket| ticket.started.elapsed() < PRELOAD_LIFETIME);
        if tickets.contains_key(&url) || tickets.len() >= MAX_PRELOADS {
            return None;
        }
        let (mut send, ready) = oneshot::channel();
        let cookies = jar.snapshot(&url);
        let claimed = Rc::new(Cell::new(false));
        tickets.insert(
            url.clone(),
            Preload {
                cookies,
                started: Instant::now(),
                ready,
                claimed: claimed.clone(),
            },
        );
        let transfer = self.download(client, jar, url, false, false);
        let state = self.inner.clone();
        let slots = self.preload_slots.clone();
        Some(Box::pin(async move {
            let expire_unused = async {
                tokio::time::sleep(PRELOAD_LIFETIME).await;
                if claimed.get() {
                    // A foreground request now owns this download, including on a slow link.
                    futures::future::pending::<()>().await;
                }
            };
            let prepare = async {
                let _slot = slots.acquire_many(weight).await.ok()?;
                transfer.await.into_preload(&state).await
            };
            tokio::select! {
                result = prepare => { let _ = send.send(result); }
                _ = send.cancellation() => {}
                _ = expire_unused => {}
            }
            Ok(())
        }))
    }

    pub(super) fn fetch(
        &self,
        client: Client,
        jar: Arc<super::SessionCookies>,
        url: Url,
        reason: FetchReason,
    ) -> OwnedFuture<Response, ErrorResponse> {
        let this = self.clone();
        Box::pin(async move {
            let preload = {
                let mut tickets = this.preloads.borrow_mut();
                tickets.retain(|_, ticket| ticket.started.elapsed() < PRELOAD_LIFETIME);
                let claimed = tickets.remove(&url);
                if this.memory.borrow().manager().pressure().level >= PressureLevel::High {
                    tickets.clear();
                }
                claimed
            };
            if let Some(ticket) = preload
                && ticket.cookies == jar.snapshot(&url)
            {
                ticket.claimed.set(true);
                if let Ok(Some(response)) = ticket.ready.await
                    && ticket.cookies == jar.snapshot(&url)
                {
                    let mut state = this.inner.borrow_mut();
                    state.stats.preloaded += 1;
                    state.stats.requested += 1;
                    return Ok(response);
                }
            }
            let _slot = this
                .admissions
                .acquire()
                .await
                .expect("session admission remains open");
            let visible = reason == FetchReason::LoadSwf
                || this.visible_assets.get().is_some_and(|f| f(&url));
            this.inner.borrow_mut().stats.requested += 1;
            if !this.enabled() || jar.cookies(&url).is_some() {
                return this
                    .download(client, jar, url, visible, false)
                    .await
                    .for_reader(true)
                    .expect("individual reader");
            }
            let (shared, leader) = {
                let mut state = this.inner.borrow_mut();

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
        jar: Arc<super::SessionCookies>,
        url: Url,
        visible: bool,
        may_share: bool,
    ) -> LocalBoxFuture<'static, Outcome> {
        let state = self.inner.clone();
        let memory = self.memory.borrow().clone();
        let vfs = self.vfs.borrow().clone();
        let private_vfs = self.account_vfs.borrow().clone();
        let disk_reads = self.disk_reads.clone();
        let schedule_transfers = self.enabled();
        let cache_first = self.cache_first.get();
        async move {
            if let Some(disk) = vfs.clone()
                && jar.cookies(&url).is_none()
            {
                let hit = read_vfs(disk, url.to_string(), disk_reads.clone()).await;
                if let Some((hit, slot)) = hit
                    && (cache_first || hit.fresh_until > unix_now())
                    && jar.cookies(&url).is_none()
                {
                    {
                        let mut s = state.borrow_mut();
                        s.stats.cache_hits += 1;
                        s.stats.cache_bytes += hit.bytes.len() as u64;
                    }
                    return Outcome::cached(&url, hit, slot, &state, &memory, may_share, true);
                }
                state.borrow_mut().stats.cache_misses += 1;
            }
            let permit = TransferPermit::acquire(
                state.clone(),
                url.origin().ascii_serialization(),
                visible,
                schedule_transfers,
            )
            .await;
            let began = Instant::now();
            let (anonymous, version) = jar.snapshot(&url);
            let store = if anonymous { vfs } else { private_vfs };
            let mut validator = None;
            if !anonymous && let Some(disk) = &store {
                let disk = disk.clone();
                let key = url.to_string();
                validator = tokio::task::spawn_blocking(move || disk.validator(&key))
                    .await
                    .ok()
                    .flatten();
            }
            let mut http = loop {
                let mut get = client.get(url.clone()).header(header::CONTENT_TYPE, "");
                if let Some(tag) = &validator {
                    get = get.header(header::IF_NONE_MATCH, tag);
                }
                let sent = run_io(get.send())
                    .await
                    .map_err(|e| (false, e.to_string()))
                    .and_then(|r| r.map_err(|e| (e.is_connect(), e.to_string())));
                let response = match sent {
                    Ok(r) => r,
                    Err((connect, message)) => {
                        permit.complete();
                        return Outcome::Failure(Failure::Request(
                            url.to_string(),
                            connect,
                            message,
                        ));
                    }
                };
                if response.status() != reqwest::StatusCode::NOT_MODIFIED || validator.is_none() {
                    break response;
                }
                if response.url() == &url
                    && public_headers(response.headers())
                    && response.headers().get(header::ETAG).is_none_or(|new| {
                        new.as_bytes() == validator.as_deref().unwrap().as_bytes()
                    })
                    && let Some(disk) = store.clone()
                    && let Some((hit, slot)) =
                        read_vfs(disk, url.to_string(), disk_reads.clone()).await
                    && hit.etag == validator
                {
                    let mut st = state.borrow_mut();
                    st.stats.cache_hits += 1;
                    st.stats.revalidated += 1;
                    st.stats.cache_bytes += hit.bytes.len() as u64;
                    st.stats.headers_ms += began.elapsed().as_millis() as u64;
                    drop(st);
                    permit.complete();
                    return Outcome::cached(&url, hit, slot, &state, &memory, false, false);
                }
                // A bad or evicted entity needs a second, unconditional network transfer.
                state.borrow_mut().stats.transfers += 1;
                validator = None;
            };
            if !anonymous && store.is_some() {
                state.borrow_mut().stats.cache_misses += 1;
            }
            state.borrow_mut().stats.headers_ms += began.elapsed().as_millis() as u64;
            let mut meta = Meta::from_http(&http, &url);
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
            let same_jar = !anonymous || jar.snapshot(&url) == (true, version);
            let cacheable = same_jar
                && http.status() == reqwest::StatusCode::OK
                && meta.url == url.as_str()
                && public_headers(http.headers());
            meta.allows_movie_cache = cacheable && anonymous && store.is_some();
            let tag = http
                .headers()
                .get(header::ETAG)
                .and_then(|h| h.to_str().ok())
                .filter(|v| v.len() <= 512 && v.starts_with('"') && v.ends_with('"'))
                .map(str::to_owned);
            let fresh_until = fresh_until(http.headers());
            // Versioned public assets are safe to retain even when the server omits
            // freshness metadata. `cache_first` can then serve that immutable URL
            // without making another network request.
            let disk = if cacheable && (anonymous || tag.is_some()) {
                store.map(|store| DiskWrite {
                    store,
                    url: url.to_string(),
                    encoding: meta.encoding.map(|e| e.name()),
                    etag: tag,
                    fresh_until,
                    state: state.clone(),
                })
            } else {
                None
            };
            let budget = if anonymous && cacheable && may_share {
                size.filter(|len| *len <= MAX_BODY)
                    .and_then(|len| Reservation::new(&state, len))
            } else {
                None
            };
            let mut prefix = MemoryBuffer::new(&memory, MemoryKind::Network, Vec::new());
            let expected = http.content_length();
            if let Some(budget) = budget
                && let Some(buffer) =
                    MemoryBuffer::try_with_capacity(&memory, MemoryKind::Network, budget.bytes)
            {
                let began = Instant::now();
                let size = budget.bytes;
                let read = run_io(read_body(http, buffer, size))
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
                            let data = AssetBytes::new(data);
                            let data = if let Some(write) = disk {
                                write.save(data).await
                            } else {
                                data
                            };
                            return Outcome::Shared(
                                meta,
                                Rc::new(SharedBody::new(data, budget, &state)),
                            );
                        }
                    }
                    Err(message) => {
                        permit.complete();
                        return Outcome::BodyError(meta, message);
                    }
                }
            }
            Outcome::Single(RefCell::new(Some(Response {
                allows_movie_cache: meta.allows_movie_cache,
                url: meta.url,
                loader_request: None,
                text_encoding: meta.encoding,
                status: meta.status,
                redirected: meta.redirected,
                response_body: ResponseBody::Scheduled(Some(UnsharedBody {
                    http,
                    prefix,
                    expected,
                    permit,
                    disk,
                })),
            })))
        }
        .boxed_local()
    }
}

// A SQLite read can continue after its waiting future is cancelled. Hold the slot in
// the blocking task and then with the result until the loader receives its bytes.
async fn read_vfs(
    store: Arc<dyn AssetVfs>,
    key: String,
    slots: Arc<tokio::sync::Semaphore>,
) -> Option<(CachedAsset, tokio::sync::OwnedSemaphorePermit)> {
    let slot = slots.acquire_owned().await.ok()?;
    tokio::task::spawn_blocking(move || store.get(&key).map(|entry| (entry, slot)))
        .await
        .ok()
        .flatten()
}
fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
fn fresh_until(headers: &header::HeaderMap) -> i64 {
    let age = headers
        .get(header::AGE)
        .and_then(|v| v.to_str().ok()?.parse::<u64>().ok())
        .unwrap_or(0);
    let ttl = headers
        .get_all(header::CACHE_CONTROL)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .filter_map(|part| {
            let (name, value) = part.trim().split_once('=')?;
            name.eq_ignore_ascii_case("max-age")
                .then(|| value.trim().trim_matches('"').parse::<u64>().ok())
                .flatten()
        })
        .min()
        .unwrap_or(0)
        .saturating_sub(age)
        .min(30 * 86400);
    unix_now() + ttl as i64
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
    async fn acquire(
        state: Rc<RefCell<State>>,
        origin: String,
        visible: bool,
        scheduled: bool,
    ) -> Self {
        let since = Instant::now();
        if scheduled {
            let id = {
                let mut s = state.borrow_mut();
                s.seq += 1;
                s.seq
            };
            let _queued = Queued {
                state: state.clone(),
                id,
            };
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
                        return Poll::Ready(());
                    }
                }
                Poll::Pending
            })
            .await;
        }
        {
            let mut s = state.borrow_mut();
            s.active += 1;
            *s.origins.entry(origin.clone()).or_default() += 1;
            s.stats.transfers += 1;
            s.stats.peak_active = s.stats.peak_active.max(s.active);
            s.stats.queue_ms += since.elapsed().as_millis() as u64;
            for w in &s.waiting {
                w.waker.wake_by_ref();
            }
        }
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
    used: Arc<AtomicUsize>,
    bytes: usize,
}
impl Reservation {
    fn new(state: &Rc<RefCell<State>>, bytes: usize) -> Option<Self> {
        let mut s = state.borrow_mut();
        if bytes > MAX_SHARED - s.shared_bytes.load(Ordering::Relaxed) {
            return None;
        }
        let used = s.shared_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        s.stats.peak_shared_bytes = s.stats.peak_shared_bytes.max(used);
        Some(Self {
            used: s.shared_bytes.clone(),
            bytes,
        })
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

pub(super) struct SharedBody {
    data: AssetBytes,
    state: Rc<RefCell<State>>,
}
impl SharedBody {
    fn new(mut data: AssetBytes, budget: Reservation, state: &Rc<RefCell<State>>) -> Self {
        data.credit = Some(Arc::new(budget));
        Self {
            data,
            state: state.clone(),
        }
    }
    pub fn into_data(self: Rc<Self>) -> AssetBytes {
        self.state.borrow_mut().stats.delivered += 1;
        match Rc::try_unwrap(self) {
            Ok(body) => body.data,
            Err(shared) => shared.data.clone(),
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn reserved_bytes(&self) -> usize {
        self.data.credit.as_ref().map_or(0, |credit| credit.bytes)
    }
}

/// Disk writers and joined readers can outlive the scheduler's response. The underlying allocation
/// owns its charge across those handoffs. A sole Vec consumer can take the original allocation.
#[derive(Clone)]
pub(super) struct AssetBytes {
    data: Arc<MemoryBuffer<u8>>,
    credit: Option<Arc<Reservation>>,
}
impl AssetBytes {
    pub(super) fn new(buffer: MemoryBuffer<u8>) -> Self {
        Self {
            data: Arc::new(buffer),
            credit: None,
        }
    }
}

impl std::ops::Deref for AssetBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.data
    }
}
impl AsRef<[u8]> for AssetBytes {
    fn as_ref(&self) -> &[u8] {
        self
    }
}
impl From<AssetBytes> for Bytes {
    fn from(value: AssetBytes) -> Self {
        Self::from_owner(value)
    }
}
impl From<AssetBytes> for Vec<u8> {
    fn from(value: AssetBytes) -> Self {
        match Arc::try_unwrap(value.data) {
            Ok(buffer) => buffer.into_vec(),
            Err(buffer) => buffer.to_vec(),
        }
    }
}
struct DiskWrite {
    store: Arc<dyn AssetVfs>,
    url: String,
    encoding: Option<&'static str>,
    etag: Option<String>,
    fresh_until: i64,
    state: Rc<RefCell<State>>,
}
impl DiskWrite {
    async fn save(self, data: AssetBytes) -> AssetBytes {
        if data.is_empty() || data.len() > 32 * 1024 * 1024 {
            return data;
        }
        // Keep the downloaded bytes usable even if the store is unavailable.
        let worker = data.clone();
        let wrote = tokio::task::spawn_blocking(move || {
            self.store.put(
                &self.url,
                self.encoding,
                self.etag.as_deref(),
                self.fresh_until,
                &worker,
            )
        })
        .await
        .unwrap_or(false);
        if wrote {
            self.state.borrow_mut().stats.cache_writes += 1;
        }
        data
    }
}

pub(super) struct UnsharedBody {
    disk: Option<DiskWrite>,
    http: reqwest::Response,
    prefix: MemoryBuffer<u8>,
    pub expected: Option<u64>,
    permit: TransferPermit,
}
impl UnsharedBody {
    pub async fn read(self) -> Result<AssetBytes, Error> {
        let state = self.permit.state.clone();
        let bytes = self.buffer().await?;
        state.borrow_mut().stats.delivered += 1;
        Ok(bytes)
    }

    async fn buffer(self) -> Result<AssetBytes, Error> {
        let began = Instant::now();
        let prefix_bytes = self.prefix.len();
        let result = run_io(read_body(self.http, self.prefix, usize::MAX)).await;
        self.permit.complete();
        let (data, _) = result
            .map_err(|e| Error::FetchError(e.to_string()))?
            .map_err(|e| Error::FetchError(e.to_string()))?;
        {
            let mut state = self.permit.state.borrow_mut();
            state.stats.body_bytes += (data.len() - prefix_bytes) as u64;
            state.stats.body_ms += began.elapsed().as_millis() as u64;
        }
        let data = AssetBytes::new(data);
        let data = if let Some(write) = self.disk {
            write.save(data).await
        } else {
            data
        };
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

pub(super) async fn read_body(
    mut response: reqwest::Response,
    mut data: MemoryBuffer<u8>,
    limit: usize,
) -> Result<(MemoryBuffer<u8>, Option<reqwest::Response>), reqwest::Error> {
    while let Some(chunk) = response.chunk().await? {
        data.extend_from_slice(&chunk);
        if data.len() > limit {
            return Ok((data, Some(response)));
        }
    }
    Ok((data, None))
}

struct Meta {
    allows_movie_cache: bool,
    url: String,
    status: u16,
    redirected: bool,
    encoding: Option<&'static Encoding>,
}
impl Meta {
    fn from_http(http: &reqwest::Response, original: &Url) -> Self {
        Self {
            allows_movie_cache: false,
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
    fn response(&self, response_body: ResponseBody) -> Response {
        Response {
            allows_movie_cache: self.allows_movie_cache,
            url: self.url.clone(),
            loader_request: None,
            status: self.status,
            redirected: self.redirected,
            text_encoding: self.encoding,
            response_body,
        }
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
    fn cached(
        url: &Url,
        hit: CachedAsset,
        slot: tokio::sync::OwnedSemaphorePermit,
        state: &Rc<RefCell<State>>,
        memory: &MemoryScope,
        may_share: bool,
        allows_movie_cache: bool,
    ) -> Self {
        let meta = Meta {
            allows_movie_cache,
            url: url.to_string(),
            status: 200,
            redirected: false,
            encoding: hit
                .encoding
                .as_deref()
                .and_then(|e| Encoding::for_label(e.as_bytes())),
        };
        let data = AssetBytes::new(MemoryBuffer::new(memory, MemoryKind::Network, hit.bytes));
        let body = if data.len() <= MAX_BODY
            && let Some(budget) = Reservation::new(state, data.len())
        {
            let body = Rc::new(SharedBody::new(data, budget, state));
            if may_share {
                return Self::Shared(meta, body);
            }
            ResponseBody::Shared(Some(body))
        } else {
            ResponseBody::Disk(Some((data, slot)))
        };
        Self::Single(RefCell::new(Some(meta.response(body))))
    }

    async fn into_preload(self, state: &Rc<RefCell<State>>) -> Option<Response> {
        let Self::Single(response) = self else {
            return None;
        };
        let mut response = response.into_inner()?;
        if matches!(response.response_body, ResponseBody::Shared(_)) {
            return Some(response);
        }
        let size = usize::try_from(response.expected_length().ok()??).ok()?;
        let reservation = Reservation::new(state, size)?;
        let data = match response.response_body {
            ResponseBody::Disk(Some((data, _slot))) => data,
            // A DiskWrite exists only after checking origin, HTTP policy and cookies.
            ResponseBody::Scheduled(Some(body)) if body.disk.is_some() => {
                body.buffer().await.ok()?
            }
            _ => return None,
        };
        if data.len() != size {
            return None;
        }
        response.response_body =
            ResponseBody::Shared(Some(Rc::new(SharedBody::new(data, reservation, state))));
        Some(response)
    }

    fn for_reader(&self, leader: bool) -> Option<Result<Response, ErrorResponse>> {
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
            Self::Single(r) if leader => Some(Ok(r.borrow_mut().take().expect("original reader"))),
            Self::Single(_) => None,
        }
    }
}
