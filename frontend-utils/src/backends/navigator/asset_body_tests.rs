use super::super::tests::new_test_backend;
use super::*;
use ruffle_core::backend::navigator::{NavigatorBackend, Request};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

struct Disk {
    size: usize,
    reads: AtomicUsize,
    last_address: AtomicUsize,
}

impl Disk {
    fn mount(scheduler: &AssetScheduler, size: usize) -> Arc<Self> {
        let disk = Arc::new(Self {
            size,
            reads: 0.into(),
            last_address: 0.into(),
        });
        scheduler.mount_vfs(disk.clone(), disk.clone());
        disk
    }
}

impl AssetVfs for Disk {
    fn get(&self, _: &str) -> Option<CachedAsset> {
        let ordinal = self.reads.fetch_add(1, Ordering::SeqCst);
        let bytes = vec![ordinal as u8; self.size];
        self.last_address
            .store(bytes.as_ptr() as usize, Ordering::SeqCst);
        Some(CachedAsset {
            bytes,
            encoding: None,
            etag: self.validator(""),
            fresh_until: i64::MAX,
        })
    }
    fn validator(&self, _: &str) -> Option<String> {
        Some("\"version\"".into())
    }
    fn put(&self, _: &str, _: Option<&str>, _: Option<&str>, _: i64, _: &[u8]) -> bool {
        panic!("unexpected disk write")
    }
}

async fn ready<F: std::future::Future>(future: F) -> F::Output {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("reader made progress")
}

#[tokio::test]
async fn two_large_fresh_hits_wait_for_body_handoff_or_cancellation() {
    let mut backend = new_test_backend(false);
    let scheduler = backend.configure_buffered_assets(|_| true);
    let disk = Disk::mount(&scheduler, MAX_BODY + 1);
    let fetch = |path| {
        backend.fetch_buffered(
            Request::get(format!("http://localhost/{path}")),
            FetchReason::LoadSwf,
        )
    };
    let first = ready(fetch("first")).await.map_err(|e| e.error).unwrap();
    assert_eq!(first.buffered_body_len(), Some(disk.size));
    let address = disk.last_address.load(Ordering::SeqCst);
    let mut second = fetch("second");
    let first = first.body_bytes();
    assert_eq!(
        scheduler.disk_reads.available_permits(),
        0,
        "unpolled body owns the disk slot"
    );
    assert!(futures::poll!(&mut second).is_pending());
    let bytes = first.await.unwrap();
    assert_eq!(bytes.as_ptr() as usize, address);
    let mut second = ready(second).await.map_err(|e| e.error).unwrap();
    let mut third = fetch("third");
    assert!(futures::poll!(&mut third).is_pending());
    let chunk = second.next_chunk();
    drop(second);
    assert_eq!(scheduler.disk_reads.available_permits(), 0);
    assert_eq!(disk.reads.load(Ordering::SeqCst), 2);
    drop(chunk);
    let third = ready(third).await.map_err(|e| e.error).unwrap();
    drop(third);
    assert_eq!(scheduler.disk_reads.available_permits(), 1);
    assert_eq!(scheduler.stats().transfers, 0);
}

#[tokio::test]
async fn large_joined_hit_retries_after_leader_releases_body() {
    let mut backend = new_test_backend(false);
    let scheduler = backend.configure_buffered_assets(|_| true);
    let disk = Disk::mount(&scheduler, MAX_BODY + 1);
    let fetch = || {
        backend.fetch_buffered(
            Request::get("http://localhost/large".into()),
            FetchReason::LoadSwf,
        )
    };
    let slot = scheduler.disk_reads.clone().acquire_owned().await.unwrap();
    let (mut leader, mut follower) = (fetch(), fetch());
    assert!(futures::poll!(&mut leader).is_pending());
    assert!(futures::poll!(&mut follower).is_pending());
    drop(slot);
    let leader = ready(leader).await.map_err(|e| e.error).unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(100), &mut follower)
            .await
            .is_err(),
        "joined retry read a second large body"
    );
    assert_eq!(scheduler.stats().joined, 1);
    assert_eq!(scheduler.disk_reads.available_permits(), 0);
    assert_eq!(disk.reads.load(Ordering::SeqCst), 1);
    drop(leader);
    let bytes = ready(follower)
        .await
        .map_err(|e| e.error)
        .unwrap()
        .body_bytes()
        .await
        .unwrap();
    assert_eq!(bytes[0], 1);
    assert_eq!(scheduler.disk_reads.available_permits(), 1);
}

#[tokio::test]
async fn small_independent_disk_hits_have_credit_while_both_headers_are_held() {
    let mut backend = new_test_backend(false);
    let scheduler = backend.configure_buffered_assets(|_| true);
    scheduler.set_enabled(false);
    let disk = Disk::mount(&scheduler, 32);
    let fetch = |path| {
        backend.fetch_buffered(
            Request::get(format!("http://localhost/{path}")),
            FetchReason::LoadSwf,
        )
    };
    let (one, two) = ready(async { futures::join!(fetch("one"), fetch("two")) }).await;
    let one = one.map_err(|e| e.error).unwrap();
    let mut two = two.map_err(|e| e.error).unwrap();
    assert_eq!(scheduler.inner.borrow().shared_bytes, 2 * disk.size);
    drop(one);
    assert_eq!(scheduler.inner.borrow().shared_bytes, disk.size);
    let bytes = two.next_chunk().await.unwrap().unwrap();
    assert_eq!(bytes.len(), disk.size);
    assert_eq!(
        bytes.as_ptr() as usize,
        disk.last_address.load(Ordering::SeqCst)
    );
    assert_eq!(scheduler.inner.borrow().shared_bytes, 0);
    assert_eq!(scheduler.disk_reads.available_permits(), 1);
}

#[tokio::test]
async fn empty_independent_disk_hit_returns_eof_on_first_read() {
    let mut backend = new_test_backend(false);
    let scheduler = backend.configure_buffered_assets(|_| true);
    scheduler.set_enabled(false);
    Disk::mount(&scheduler, 0);
    let mut response = ready(backend.fetch_buffered(
        Request::get("http://localhost/empty".into()),
        FetchReason::LoadSwf,
    ))
    .await
    .map_err(|e| e.error)
    .unwrap();
    assert_eq!(response.buffered_body_len(), Some(0));
    assert!(response.next_chunk().await.unwrap().is_none());
    assert_eq!(scheduler.disk_reads.available_permits(), 1);
}

// Flush real headers and let the test decide when the server supplies (or truncates) the body.
async fn serve(
    headers: String,
    conditional: bool,
) -> (Url, oneshot::Sender<Vec<u8>>, tokio::task::JoinHandle<()>) {
    let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = Url::parse(&format!("http://{}/asset", server.local_addr().unwrap())).unwrap();
    let (send, body) = oneshot::channel::<Vec<u8>>();
    let task = tokio::spawn(async move {
        let (mut stream, _) = server.accept().await.unwrap();
        let mut lines = BufReader::new(&mut stream).lines();
        assert_eq!(
            lines.next_line().await.unwrap().as_deref(),
            Some("GET /asset HTTP/1.1")
        );
        let mut validated = false;
        while let Some(line) = lines.next_line().await.unwrap() {
            if line.is_empty() {
                break;
            }
            validated |= line.eq_ignore_ascii_case("If-None-Match: \"version\"");
        }
        assert_eq!(validated, conditional);
        stream.write_all(headers.as_bytes()).await.unwrap();
        if let Ok(data) = body.await {
            stream.write_all(&data).await.unwrap();
        }
    });
    (url, send, task)
}

#[tokio::test]
async fn validated_large_body_keeps_disk_credit_after_http_completes() {
    let mut backend = new_test_backend(false);
    let scheduler = backend.configure_buffered_assets(|_| true);
    let disk = Disk::mount(&scheduler, MAX_BODY + 1);
    let (url, send, server) = serve("HTTP/1.1 304 Not Modified\r\n\r\n".into(), true).await;
    backend
        .cookies
        .jar
        .add_cookie_str("account=test; Path=/", &url);
    let mut response =
        ready(backend.fetch_buffered(Request::get(url.into()), FetchReason::LoadSwf))
            .await
            .map_err(|e| e.error)
            .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.expected_length().unwrap(), Some(disk.size as u64));
    assert_eq!(response.buffered_body_len(), Some(disk.size));
    assert_eq!(scheduler.stats().revalidated, 1);
    assert_eq!(scheduler.inner.borrow().active, 0);
    assert_eq!(scheduler.disk_reads.available_permits(), 0);
    let bytes = response.next_chunk().await.unwrap().unwrap();
    assert_eq!(
        bytes.as_ptr() as usize,
        disk.last_address.load(Ordering::SeqCst)
    );
    assert!(response.next_chunk().await.unwrap().is_none());
    assert_eq!(scheduler.disk_reads.available_permits(), 1);
    drop(send);
    ready(server).await.unwrap();
}

#[tokio::test]
async fn oversized_http_body_keeps_its_transfer_until_handoff() {
    let mut backend = new_test_backend(false);
    let scheduler = backend.configure_buffered_assets(|_| true);
    let size = MAX_BODY + 1;
    let (url, finish, server) = serve(
        format!("HTTP/1.1 200 OK\r\nContent-Length: {size}\r\n\r\n"),
        false,
    )
    .await;
    let response = ready(backend.fetch_buffered(Request::get(url.into()), FetchReason::LoadSwf))
        .await
        .map_err(|e| e.error)
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.buffered_body_len(), None);
    assert_eq!(response.expected_length().unwrap(), Some(size as u64));
    let mut body = response.body_bytes();
    assert!(futures::poll!(&mut body).is_pending());
    assert_eq!(scheduler.inner.borrow().active, 1);
    finish.send(vec![0xa5; size]).unwrap();
    let bytes = ready(body).await.unwrap();
    assert_eq!(bytes.len(), size);
    assert!(bytes.iter().all(|&b| b == 0xa5));
    assert_eq!(scheduler.inner.borrow().active, 0);
    assert_eq!(scheduler.stats().delivered, 1);
    ready(server).await.unwrap();
}

#[tokio::test]
async fn successful_http_headers_keep_transfer_credit_through_body_success_error_and_drop() {
    let mut backend = new_test_backend(false);
    let scheduler = backend.configure_buffered_assets(|_| true);
    for (headers, expected, payload) in [
        (
            "Transfer-Encoding: chunked".into(),
            None,
            Some(&b"2\r\nok\r\n0\r\n\r\n"[..]),
        ),
        (
            format!("Content-Length: {}", MAX_BODY + 1),
            Some((MAX_BODY + 1) as u64),
            Some(&b"short"[..]),
        ),
        ("Transfer-Encoding: chunked".into(), None, None),
    ] {
        let (url, send, server) =
            serve(format!("HTTP/1.1 200 OK\r\n{headers}\r\n\r\n"), false).await;
        let response =
            ready(backend.fetch_buffered(Request::get(url.into()), FetchReason::LoadSwf))
                .await
                .map_err(|e| e.error)
                .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.expected_length().unwrap(), expected);
        assert_eq!(response.buffered_body_len(), None);
        assert_eq!(scheduler.inner.borrow().active, 1);
        let mut body = response.body_bytes();
        assert!(futures::poll!(&mut body).is_pending());
        assert_eq!(scheduler.inner.borrow().active, 1);
        if let Some(payload) = payload {
            send.send(payload.to_vec()).unwrap();
            let result = ready(body).await;
            if expected.is_some() {
                assert!(matches!(result, Err(Error::FetchError(_))));
            } else {
                assert_eq!(result.unwrap().as_ref(), b"ok");
            }
        } else {
            drop(body);
            drop(send);
        }
        ready(server).await.unwrap();
        assert_eq!(scheduler.inner.borrow().active, 0);
        assert!(scheduler.inner.borrow().origins.is_empty());
    }
    assert_eq!(scheduler.stats().cancelled, 1);
    assert_eq!(scheduler.stats().delivered, 1);
}
