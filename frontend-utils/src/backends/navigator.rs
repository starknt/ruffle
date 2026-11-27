mod assets;
mod fetch;
pub use assets::{AssetScheduler, AssetStats, AssetVfs, CachedAsset};

use crate::backends::navigator::fetch::{Response, ResponseBody};
use crate::content::PlayingContent;
use async_channel::Receiver;
use async_io::Timer;
use futures_lite::FutureExt;
use reqwest::{Proxy, cookie, cookie::CookieStore, header};
use ruffle_core::backend::navigator::{
    ErrorResponse, NavigationMethod, NavigatorBackend, OwnedFuture, Request, SocketMode,
    SuccessResponse, async_return, create_fetch_error, get_encoding,
};
use ruffle_core::indexmap::IndexMap;
use ruffle_core::loader::Error;
use ruffle_core::socket::{
    ConnectionState, SocketAction, SocketActionSender, SocketHandle, SocketWrite,
};
use std::collections::HashSet;
use std::fs::File;
use std::future::Future;
use std::io;
use std::io::ErrorKind;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::warn;
use url::{ParseError, Url};

/// Keep a flushed payload and its byte credit together through partial writes.
async fn write_socket_payloads(
    writer: &mut (impl AsyncWrite + Unpin),
    receiver: Receiver<SocketWrite>,
) -> io::Result<()> {
    while let Ok(payload) = receiver.recv().await {
        let mut remaining = payload.as_ref();
        while !remaining.is_empty() {
            match writer.write(remaining).await {
                Ok(0) => return Err(ErrorKind::WriteZero.into()),
                Ok(written) => remaining = &remaining[written..],
                Err(error)
                    if matches!(error.kind(), ErrorKind::TimedOut | ErrorKind::Interrupted) => {}
                Err(error) => return Err(error),
            }
        }
    }
    Ok(())
}

/// The version binds admission to the cookie state that reqwest actually used,
/// including cookies set while an HTTP request was awaiting its connection.
#[derive(Default)]
pub(super) struct SessionCookies {
    jar: cookie::Jar,
    revision: std::sync::RwLock<u64>,
}
impl SessionCookies {
    fn snapshot(&self, url: &Url) -> (bool, u64) {
        let revision = self.revision.read().expect("cookie version lock");
        (self.jar.cookies(url).is_none(), *revision)
    }
}
impl CookieStore for SessionCookies {
    fn set_cookies(&self, headers: &mut dyn Iterator<Item = &header::HeaderValue>, url: &Url) {
        let mut headers = headers.peekable();
        if headers.peek().is_none() {
            return;
        }
        let mut version = self.revision.write().expect("cookie version lock");
        self.jar.set_cookies(&mut headers, url);
        *version = version.wrapping_add(1);
    }
    fn cookies(&self, url: &Url) -> Option<header::HeaderValue> {
        self.jar.cookies(url)
    }
}

pub trait NavigatorInterface: Clone + Send + 'static {
    fn navigate_to_website(&self, url: Url);

    fn open_file(&self, path: &Path) -> impl std::future::Future<Output = io::Result<File>> + Send;

    fn confirm_socket(
        &self,
        host: &str,
        port: u16,
    ) -> impl std::future::Future<Output = bool> + Send;
}

use ruffle_core::backend::tasks::TaskSpawner;

/// Loads resources on the native host and opens links through its system browser.
pub struct ExternalNavigatorBackend<F: TaskSpawner, I: NavigatorInterface> {
    /// Schedules transport-owned asset preloads on the host event loop.
    future_spawner: F,

    /// The url to use for all relative fetches.
    base_url: Url,

    // Client to use for network requests
    client: Option<Rc<reqwest::Client>>,

    cookies: Arc<SessionCookies>,
    assets: Option<AssetScheduler>,

    socket_allowed: HashSet<String>,

    socket_mode: SocketMode,

    upgrade_to_https: bool,

    content: Rc<PlayingContent>,

    interface: I,
}

impl<F: TaskSpawner, I: NavigatorInterface> ExternalNavigatorBackend<F, I> {
    /// Scope public whole-body reuse and connection limits to this navigator.
    pub fn configure_buffered_assets(&mut self, allow: fn(&Url) -> bool) -> AssetScheduler {
        let scheduler = AssetScheduler::new(allow);
        self.assets = Some(scheduler.clone());
        scheduler
    }

    /// Reqwest exposes its connection pool through requests. A bodyless HEAD
    /// makes DNS and TLS overlap renderer startup, including on a hot disk cache.
    pub fn warm_connection(
        &self,
        public_asset: &Url,
    ) -> impl Future<Output = ()> + Send + use<F, I> {
        let head = self.client.as_ref().map(|client| {
            client
                .head(public_asset.clone())
                .timeout(Duration::from_secs(2))
        });
        async move {
            if let Some(head) = head {
                let _ = head.send().await;
            }
        }
    }

    /// Warm an allow-listed static asset on the same HTTP pool used by the movie.
    pub fn preload(&self, url: &Url) {
        if matches!(url.scheme(), "https" | "http")
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
            && let (Some(scheduler), Some(client)) = (&self.assets, &self.client)
            && let Some(future) =
                scheduler.preload(client.as_ref().clone(), self.cookies.clone(), url.clone())
        {
            self.future_spawner.spawn(future);
        }
    }

    /// Construct a navigator backend with fetch and async capability.
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        mut base_url: Url,
        referer: Option<Url>,
        cookie: Option<String>,
        future_spawner: F,
        proxy: Option<Url>,
        upgrade_to_https: bool,
        socket_allowed: HashSet<String>,
        socket_mode: SocketMode,
        content: Rc<PlayingContent>,
        interface: I,
    ) -> Self {
        let cookies = Arc::new(SessionCookies::default());
        let mut builder = reqwest::ClientBuilder::new()
            .cookie_provider(cookies.clone())
            .user_agent(concat!(
                "Ruffle/",
                env!("CARGO_PKG_VERSION"),
                " (https://ruffle.rs)"
            ));

        if let Some(referer) = referer {
            let mut headers = header::HeaderMap::new();
            headers.insert(header::REFERER, referer.to_string().parse().unwrap());
            builder = builder.default_headers(headers);
        }

        if let Some(cookie) = cookie {
            cookies.jar.add_cookie_str(&cookie, &base_url);
        }

        if let Some(proxy) = proxy {
            match Proxy::all(proxy.clone()) {
                Ok(proxy) => {
                    builder = builder.proxy(proxy);
                }
                Err(e) => {
                    tracing::error!("Couldn't configure proxy {proxy}: {e}")
                }
            }
        }

        let client = builder.build().ok().map(Rc::new);

        // Force replace the last segment with empty. //

        if let Ok(mut base_url) = base_url.path_segments_mut() {
            base_url.pop().pop_if_empty().push("");
        }

        Self {
            future_spawner,
            cookies,
            assets: None,
            client,
            base_url,
            upgrade_to_https,
            socket_allowed,
            socket_mode,
            content,
            interface,
        }
    }
}

impl<F: TaskSpawner + 'static, I: NavigatorInterface> NavigatorBackend
    for ExternalNavigatorBackend<F, I>
{
    fn navigate_to_url(
        &self,
        url: &str,
        _target: &str,
        vars_method: Option<(NavigationMethod, IndexMap<String, String>)>,
    ) {
        //TODO: Should we return a result for failed opens? Does Flash care?

        //NOTE: Flash desktop players / projectors ignore the window parameter,
        //      unless it's a `_layer`, and we shouldn't handle that anyway.
        let mut parsed_url = match self.resolve_url(url) {
            Ok(parsed_url) => parsed_url,
            Err(e) => {
                tracing::error!(
                    "Could not parse URL because of {}, the corrupt URL was: {}",
                    e,
                    url
                );
                return;
            }
        };

        let modified_url = match vars_method {
            Some((_, query_pairs)) if !query_pairs.is_empty() => {
                {
                    //lifetime limiter because we don't have NLL yet
                    let mut modifier = parsed_url.query_pairs_mut();

                    for (k, v) in query_pairs.iter() {
                        modifier.append_pair(k, v);
                    }
                }

                parsed_url
            }
            _ => parsed_url,
        };

        if modified_url.scheme() == "javascript" {
            tracing::warn!(
                "SWF tried to run a script on desktop, but javascript calls are not allowed"
            );
            return;
        }

        self.interface.navigate_to_website(modified_url);
    }

    fn fetch(&self, request: Request) -> OwnedFuture<Box<dyn SuccessResponse>, ErrorResponse> {
        // TODO: honor sandbox type (local-with-filesystem, local-with-network, remote, ...)
        let mut processed_url = match self.resolve_url(request.url()) {
            Ok(url) => url,
            Err(e) => {
                return async_return(Err(create_fetch_error(request.url(), e)));
            }
        };

        let client = self.client.clone();
        let loader_request = request.loader_request();

        match processed_url.scheme() {
            "file" => {
                let content = self.content.clone();
                let interface = self.interface.clone();
                Box::pin(async move {
                    // We send the original url (including query parameters)
                    // back to ruffle_core in the `Response`
                    let response_url = processed_url.clone();
                    // Flash supports query parameters with local urls.
                    // SwfMovie takes care of exposing those to ActionScript -
                    // when we actually load a filesystem url, strip them out.
                    processed_url.set_query(None);

                    let contents = content.get_local_file(&processed_url, interface).await;

                    let response: Box<dyn SuccessResponse> = Box::new(Response {
                        url: response_url.to_string(),
                        loader_request,
                        response_body: ResponseBody::File(contents.map(Into::into)),
                        text_encoding: None,
                        status: 0,
                        redirected: false,
                    });

                    Ok(response)
                })
            }
            _ => Box::pin(async move {
                let client = client.ok_or_else(|| ErrorResponse {
                    url: processed_url.to_string(),
                    error: Error::FetchError("Network unavailable".to_string()),
                })?;

                let mut request_builder = match request.method() {
                    NavigationMethod::Get => client.get(processed_url.clone()),
                    NavigationMethod::Post => client.post(processed_url.clone()),
                };
                let (body_data, mime) = request.body().clone().unwrap_or_default();
                for (name, val) in request.headers().iter() {
                    request_builder = request_builder.header(name, val);
                }
                request_builder = request_builder.header("Content-Type", &mime);

                request_builder = request_builder.body(body_data);

                let response = spawn_tokio(request_builder.send()).await.map_err(|e| {
                    let inner = if e.is_connect() {
                        Error::InvalidDomain(processed_url.to_string())
                    } else {
                        Error::FetchError(e.to_string())
                    };
                    ErrorResponse {
                        url: processed_url.to_string(),
                        error: inner,
                    }
                })?;

                let url = response.url().to_string();
                let text_encoding = response
                    .headers()
                    .get("Content-Type")
                    .and_then(|content_type| content_type.to_str().ok())
                    .and_then(get_encoding);
                let status = response.status().as_u16();
                let redirected = *response.url() != processed_url;
                if !response.status().is_success() {
                    let error = Error::HttpNotOk(
                        format!("Got {}", response.status()),
                        status,
                        redirected,
                        response.content_length().unwrap_or_default(),
                    );
                    return Err(ErrorResponse { url, error });
                }

                let response: Box<dyn SuccessResponse> = Box::new(Response {
                    url,
                    loader_request,
                    response_body: ResponseBody::Network(Arc::new(Mutex::new(Some(response)))),
                    text_encoding,
                    status,
                    redirected,
                });
                Ok(response)
            }),
        }
    }

    fn fetch_buffered(
        &self,
        request: Request,
        reason: ruffle_core::backend::navigator::FetchReason,
    ) -> OwnedFuture<Box<dyn SuccessResponse>, ErrorResponse> {
        if let (Some(scheduler), Some(client), Ok(url)) =
            (&self.assets, &self.client, self.resolve_url(request.url()))
        {
            let eligible = request.method() == NavigationMethod::Get
                && request.body().is_none()
                && request.headers().is_empty()
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
                && scheduler.allows(&url);
            let cookie = eligible && self.cookies.cookies(&url).is_some();
            scheduler.observe(eligible, cookie);
            if eligible && (!cookie || scheduler.cache_enabled()) {
                let loader_request = request.loader_request();
                let fetch =
                    scheduler.fetch(client.as_ref().clone(), self.cookies.clone(), url, reason);
                return Box::pin(async move {
                    let mut response = fetch.await?;
                    // Joined transfers and preloads acquire their own tag at logical delivery.
                    response.loader_request = loader_request;
                    Ok(Box::new(response) as Box<dyn SuccessResponse>)
                });
            }
        }
        self.fetch(request)
    }

    fn resolve_url(&self, url: &str) -> Result<Url, ParseError> {
        match self.base_url.join(url) {
            Ok(url) => Ok(self.pre_process_url(url)),
            Err(error) => Err(error),
        }
    }

    fn pre_process_url(&self, mut url: Url) -> Url {
        if self.upgrade_to_https && url.scheme() == "http" && url.set_scheme("https").is_err() {
            tracing::error!("Url::set_scheme failed on: {}", url);
        }
        url
    }

    fn connect_socket(
        &mut self,
        host: String,
        port: u16,
        timeout: Duration,
        handle: SocketHandle,
        receiver: Receiver<SocketWrite>,
        sender: SocketActionSender,
    ) {
        /// Tries to send the given action properly handling failures.
        ///
        /// Returns `true` when the action has been sent properly,
        /// `false` when the channel is closed.
        async fn send_action(sender: &SocketActionSender, action: SocketAction) -> bool {
            sender
                .send(action)
                .await
                .inspect_err(|err| tracing::warn!("Failed to send SocketAction: {}", err))
                .is_ok()
        }

        let addr = format!("{host}:{port}");
        let is_allowed = self.socket_allowed.contains(&addr);
        let socket_mode = self.socket_mode;
        let interface = self.interface.clone();

        let future = Box::pin(async move {
            match (is_allowed, socket_mode) {
                (false, SocketMode::Allow) | (true, _) => {} // the process is allowed to continue. just dont do anything.
                (false, SocketMode::Deny) => {
                    // Just fail the connection.
                    let action = SocketAction::Connect(handle, ConnectionState::Failed);
                    let _ = send_action(&sender, action).await;

                    tracing::warn!(
                        "SWF tried to open a socket, but opening a socket is not allowed"
                    );

                    return;
                }
                (false, SocketMode::Ask) => {
                    let attempt_sandbox_connect = interface.confirm_socket(&host, port).await;

                    if !attempt_sandbox_connect {
                        // fail the connection.
                        let action = SocketAction::Connect(handle, ConnectionState::Failed);
                        let _ = send_action(&sender, action).await;
                        return;
                    }
                }
            }

            let host2 = host.clone();

            let timeout = async {
                Timer::after(timeout).await;
                Result::<TcpStream, io::Error>::Err(io::Error::new(ErrorKind::TimedOut, ""))
            };

            let mut stream = match TcpStream::connect((host, port)).or(timeout).await {
                Err(e) if e.kind() == ErrorKind::TimedOut => {
                    warn!("Connection to {}:{} timed out", host2, port);
                    let action = SocketAction::Connect(handle, ConnectionState::TimedOut);
                    let _ = send_action(&sender, action).await;
                    return;
                }
                Ok(stream) => {
                    let action = SocketAction::Connect(handle, ConnectionState::Connected);
                    if !send_action(&sender, action).await {
                        return;
                    }
                    stream
                }
                Err(err) => {
                    warn!("Failed to connect to {}:{}, error: {}", host2, port, err);
                    let action = SocketAction::Connect(handle, ConnectionState::Failed);
                    let _ = send_action(&sender, action).await;
                    return;
                }
            };

            let sender = sender;
            //NOTE: We clone the sender here as we cant share it between async tasks.
            let sender2 = sender.clone();
            let (mut read, mut write) = stream.split();

            let read = async move {
                loop {
                    let mut buffer = [0; 4096];

                    match read.read(&mut buffer).await {
                        Err(e) if e.kind() == ErrorKind::TimedOut => {} // try again later.
                        Err(_) | Ok(0) => {
                            let _ = send_action(&sender, SocketAction::Close(handle)).await;
                            break;
                        }
                        Ok(read) => {
                            let buffer = buffer.into_iter().take(read).collect::<Vec<_>>();

                            let action = SocketAction::Data(handle, buffer);
                            if !send_action(&sender, action).await {
                                return;
                            }
                        }
                    };
                }
            };

            let write = async move {
                if write_socket_payloads(&mut write, receiver).await.is_err() {
                    let _ = send_action(&sender2, SocketAction::Close(handle)).await;
                }
            };

            //NOTE: If one future exits, this will take the other one down too.
            tokio::select! {
               _ = read => {},
               _ = write => {},
            }

            if let Err(e) = stream.shutdown().await {
                tracing::warn!("Failed to shutdown write half of TcpStream: {e}");
            }
        });

        tokio::spawn(future);
    }
}

/// Runs a future on Tokio until it completes or its caller drops the returned future.
pub async fn spawn_tokio<F>(future: F) -> F::Output
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let (task, result) = futures::FutureExt::remote_handle(future);
    tokio::spawn(task);
    result.await
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use ruffle_core::socket::SocketAction::{Close, Connect, Data};
    use std::net::SocketAddr;
    use std::str::FromStr;
    use tokio::net::TcpListener;
    use tokio::task;

    use crate::content::ContentDescriptor;

    use super::*;

    impl NavigatorInterface for () {
        fn navigate_to_website(&self, _url: Url) {}

        async fn open_file(&self, path: &Path) -> io::Result<File> {
            File::open(path)
        }

        async fn confirm_socket(&self, _host: &str, _port: u16) -> bool {
            true
        }
    }

    const TIMEOUT_ZERO: Duration = Duration::ZERO;
    // The timeout has to be large enough to allow "instantaneous" actions
    // and local IO to execute, but small enough to fail tests quickly.
    const TIMEOUT: Duration = Duration::from_secs(1);

    pub(super) struct TestFutureSpawner;

    impl TaskSpawner for TestFutureSpawner {
        fn try_spawn(
            &self,
            future: ruffle_core::backend::tasks::PlayerTask,
        ) -> Result<(), ruffle_core::backend::tasks::SpawnError> {
            task::spawn_local(future);
            Ok(())
        }
    }

    macro_rules! async_timeout {
        () => {
            async {
                Timer::after(TIMEOUT).await;
                panic!("An action which should complete timed out")
            }
        };
    }

    macro_rules! async_test {
        (
            async fn $test_name:ident() $content:block
        ) => {
            #[tokio::test(flavor = "current_thread")]
            async fn $test_name() {
                task::LocalSet::new().run_until(async move $content).await;
            }
        }
    }

    macro_rules! dummy_handle {
        () => {
            SocketHandle::default()
        };
    }

    macro_rules! assert_next_socket_actions {
        ($receiver:expr;) => {
            // no more actions
        };
        ($receiver:expr; $action:expr, $($more:expr,)*) => {
            assert_eq!($receiver.recv().or(async_timeout!()).await.expect("receive action"), $action);
            assert_next_socket_actions!($receiver; $($more,)*);
        };
    }

    pub(super) fn new_test_backend(
        socket_allow: bool,
    ) -> ExternalNavigatorBackend<TestFutureSpawner, ()> {
        let url = Url::parse("https://example.com/path/").unwrap();
        ExternalNavigatorBackend::new(
            url.clone(),
            None,
            None,
            TestFutureSpawner,
            None,
            false,
            Default::default(),
            if socket_allow {
                SocketMode::Allow
            } else {
                SocketMode::Deny
            },
            Rc::new(PlayingContent::DirectFile(ContentDescriptor {
                url,
                root_content_path: None,
            })),
            (),
        )
    }

    async fn start_test_server() -> (task::JoinHandle<TcpStream>, SocketAddr) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let accept_task = task::spawn_local(async move {
            let (socket, _) = listener.accept().or(async_timeout!()).await.unwrap();
            socket
        });
        (accept_task, addr)
    }

    fn connect_test_socket(
        addr: SocketAddr,
        timeout: Duration,
        socket_allow: bool,
    ) -> (
        ruffle_core::socket::SocketOutput,
        ruffle_core::socket::SocketActionReceiver,
    ) {
        let mut backend = new_test_backend(socket_allow);

        let (write, receiver) = ruffle_core::socket::SocketOutput::channel(
            ruffle_core::socket::SocketWriteBudget::new(8192),
        );
        let (sender, read) = SocketActionSender::channel(8192);

        backend.connect_socket(
            addr.ip().to_string(),
            addr.port(),
            timeout,
            dummy_handle!(),
            receiver,
            sender,
        );

        (write, read)
    }

    async fn write_server(server_socket: &mut TcpStream, data: &str) {
        server_socket
            .write(data.as_bytes())
            .or(async_timeout!())
            .await
            .expect("server write");
    }

    async fn read_server(server_socket: &mut TcpStream) -> String {
        let mut buffer = [0; 4096];

        let read = match server_socket.read(&mut buffer).await {
            Err(e) => panic!("server read error: {e}"),
            Ok(read) => read,
        };

        let buffer = buffer.into_iter().take(read).collect::<Vec<_>>();
        String::from_utf8(buffer).unwrap()
    }

    async fn write_client(client_write: &ruffle_core::socket::SocketOutput, data: &str) {
        client_write
            .try_write(&mut data.as_bytes().to_vec())
            .expect("client write");
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn dropping_direct_fetch_before_headers_closes_the_request() {
        use tokio::io::{AsyncBufReadExt, BufReader};

        let (accept, addr) = start_test_server().await;
        let mut backend = new_test_backend(false);
        backend.client = Some(Rc::new(
            reqwest::Client::builder().no_proxy().build().unwrap(),
        ));
        let mut fetch = backend.fetch(Request::get(format!("http://{addr}/")));
        assert!(futures::poll!(&mut fetch).is_pending());

        let mut socket = accept.await.unwrap();
        let read_headers = async {
            let mut headers = BufReader::new(&mut socket).lines();
            assert_eq!(
                headers.next_line().await.unwrap().as_deref(),
                Some("GET / HTTP/1.1")
            );
            while !headers.next_line().await.unwrap().unwrap().is_empty() {}
        };
        read_headers.or(async_timeout!()).await;
        drop(fetch);
        assert_eq!(socket.read(&mut [0]).or(async_timeout!()).await.unwrap(), 0);
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn joined_readers_share_the_buffered_allocation() {
        use ruffle_core::backend::navigator::{FetchReason, LoaderRequest};
        use tokio::io::{AsyncBufReadExt, BufReader};

        let (accept, addr) = start_test_server().await;
        let serve = async {
            let mut socket = accept.await.unwrap();
            let mut headers = BufReader::new(&mut socket).lines();
            assert_eq!(
                headers.next_line().await.unwrap().as_deref(),
                Some("GET /image.png HTTP/1.1")
            );
            while !headers.next_line().await.unwrap().unwrap().is_empty() {}
            write_server(
                &mut socket,
                "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nimage",
            )
            .await;
        };
        let mut backend = new_test_backend(false);
        let scheduler = backend.configure_buffered_assets(|_| true);
        let first = LoaderRequest {
            session_generation: 7,
            request_id: 3,
            resource_version: 1,
        };
        let second = LoaderRequest {
            request_id: 4,
            resource_version: 2,
            ..first
        };
        let load = |tag| {
            let mut request = Request::get(format!("http://{addr}/image.png"));
            request.set_loader_request(tag);
            let response = backend.fetch_buffered(request, FetchReason::LoadSwf);
            async move {
                let response = response.await.map_err(|e| e.error).unwrap();
                assert_eq!(response.loader_request(), Some(tag));
                response.body_bytes().await.unwrap()
            }
        };
        let (_, first, second) = futures::join!(serve, load(first), load(second));
        assert_eq!(&first[..], b"image");
        assert_eq!(first.as_ptr(), second.as_ptr());
        let stats = scheduler.stats();
        assert_eq!(stats.joined, 1);
        assert_eq!(stats.transfers, 1);
        assert_eq!(stats.peak_shared_bytes, first.len());
        assert_eq!(stats.delivered, 2);
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_timeout() {
        let (_accept_task, addr) = start_test_server().await;
        let (_client_write, client_read) = connect_test_socket(addr, TIMEOUT_ZERO, true);
        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::TimedOut),
        );
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_connect() {
        let (accept_task, addr) = start_test_server().await;
        let (_client_write, client_read) = connect_test_socket(addr, TIMEOUT, true);
        let _server_socket = accept_task.await.unwrap();
        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Connected),
        );
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_deny() {
        let (_accept_task, addr) = start_test_server().await;
        let (_client_write, client_read) = connect_test_socket(addr, TIMEOUT, false);

        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Failed),
        );
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_fail() {
        // We want to use here an address that will cause the connection to fail instantly.
        // On Windows and macOS we use an IPv6 black hole address,
        // as connections to 127.0.0.42:226 do not fail instantly.
        // On Linux we try to use a loopback address with an unused port,
        // because some hosts try connecting to the black hole address.
        let addr = if cfg!(target_os = "linux") {
            SocketAddr::from_str("127.0.0.42:226")
        } else {
            SocketAddr::from_str("[100::]:42")
        }
        .expect("test address");
        let (_client_write, client_read) = connect_test_socket(addr, TIMEOUT, true);

        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Failed),
        );
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_server_close() {
        let (accept_task, addr) = start_test_server().await;
        let (_client_write, client_read) = connect_test_socket(addr, TIMEOUT, true);

        let server_socket = accept_task.await.unwrap();
        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Connected),
        );

        drop(server_socket);

        assert_next_socket_actions!(
            client_read;
            Close(dummy_handle!()),
        );
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_client_close() {
        let (accept_task, addr) = start_test_server().await;
        let (client_write, client_read) = connect_test_socket(addr, TIMEOUT, true);

        let mut server_socket = accept_task.await.unwrap();
        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Connected),
        );

        drop(client_write);

        assert_eq!(read_server(&mut server_socket).await, "");
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_basic_communication() {
        let (accept_task, addr) = start_test_server().await;
        let (client_write, client_read) = connect_test_socket(addr, TIMEOUT, true);

        let mut server_socket = accept_task.await.unwrap();
        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Connected),
        );

        write_server(&mut server_socket, "Hello World!").await;

        assert_next_socket_actions!(
            client_read;
            Data(dummy_handle!(), "Hello World!".as_bytes().to_vec()),
        );

        write_client(&client_write, "Hello from client").await;

        assert_eq!(read_server(&mut server_socket).await, "Hello from client");

        write_server(&mut server_socket, "from server 2").await;
        write_client(&client_write, "from client 2").await;

        assert_next_socket_actions!(
            client_read;
            Data(dummy_handle!(), "from server 2".as_bytes().to_vec()),
        );
        assert_eq!(read_server(&mut server_socket).await, "from client 2");
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn partial_socket_writes_keep_credit_and_flush_before_shutdown() {
        use ruffle_core::socket::{SocketOutput, SocketQueueError, SocketWriteBudget};

        let budget = SocketWriteBudget::new(128);
        let (out, receiver) = SocketOutput::channel(budget.clone());
        let mut bytes: Vec<_> = (0..128).collect();
        out.try_write(&mut bytes).unwrap();
        out.close();
        let (mut client, mut server) = tokio::io::duplex(7);
        let writer = task::spawn_local(async move {
            write_socket_payloads(&mut client, receiver).await.unwrap();
        });
        let mut first = [0; 1];
        server
            .read_exact(&mut first)
            .or(async_timeout!())
            .await
            .unwrap();
        assert_eq!(first, [0]);
        let (second, _reader) = SocketOutput::channel(budget);
        let mut retry = vec![99];
        assert!(matches!(
            second.try_write(&mut retry),
            Err(SocketQueueError::Full)
        ));
        let mut remaining = Vec::new();
        server
            .read_to_end(&mut remaining)
            .or(async_timeout!())
            .await
            .unwrap();
        assert_eq!(remaining, (1..128).collect::<Vec<u8>>());
        writer.await.unwrap();
        second.try_write(&mut retry).unwrap();
        assert!(retry.is_empty());
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn tcp_reader_delivers_all_bytes_before_one_remote_close() {
        let (accept, addr) = start_test_server().await;
        let (_out, inbox) = connect_test_socket(addr, TIMEOUT, true);
        let mut server = accept.await.unwrap();
        assert_next_socket_actions!(inbox; Connect(dummy_handle!(), ConnectionState::Connected),);
        let data: Vec<u8> = (0..65_536).map(|i| (i % 251) as u8).collect();
        server.write_all(&data).await.unwrap();
        server.shutdown().await.unwrap();
        let mut received = Vec::new();
        loop {
            match inbox.recv().or(async_timeout!()).await.unwrap() {
                Data(_, bytes) => received.extend(bytes),
                Close(_) => break,
                other => panic!("unexpected action {other:?}"),
            }
        }
        assert_eq!(received, data);
        assert!(inbox.recv().or(async_timeout!()).await.is_err());
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_flush_before_close() {
        let (accept_task, addr) = start_test_server().await;
        let (client_write, client_read) = connect_test_socket(addr, TIMEOUT, true);

        let mut server_socket = accept_task.await.unwrap();
        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Connected),
        );

        write_client(&client_write, "Sending some data").await;
        client_write.close();

        assert_eq!(read_server(&mut server_socket).await, "Sending some data");
    }
}
