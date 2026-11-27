use reqwest::Response as ReqwestResponse;
use ruffle_core::backend::navigator::{Bytes, LoaderRequest, OwnedFuture, SuccessResponse};
use ruffle_core::loader::Error;
use ruffle_core::swf::Encoding;
use std::sync::{Arc, Mutex};

pub(super) enum ResponseBody {
    Shared(Option<std::rc::Rc<super::assets::SharedBody>>),
    /// A cache hit holds the disk slot until a consumer takes the bytes.
    Disk(Option<(Bytes, tokio::sync::OwnedSemaphorePermit)>),
    Scheduled(Option<super::assets::UnsharedBody>),
    /// The response's body comes from a file.
    File(Result<Bytes, std::io::Error>),

    /// The response's body comes from the network.
    ///
    /// This has to be stored in shared ownership so that we can return
    /// owned futures. A synchronous lock is used here as we do not
    /// expect contention on this lock.
    Network(Arc<Mutex<Option<ReqwestResponse>>>),
}

pub(super) struct Response {
    pub url: String,
    pub loader_request: Option<LoaderRequest>,
    pub response_body: ResponseBody,
    pub text_encoding: Option<&'static Encoding>,
    pub status: u16,
    pub redirected: bool,
}

impl SuccessResponse for Response {
    fn url(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed(&self.url)
    }

    fn set_url(&mut self, url: String) {
        self.url = url;
    }

    fn loader_request(&self) -> Option<LoaderRequest> {
        self.loader_request
    }

    fn body(self: Box<Self>) -> OwnedFuture<Vec<u8>, Error> {
        Box::pin(async move { Ok(self.body_bytes().await?.into()) })
    }

    fn body_bytes(self: Box<Self>) -> OwnedFuture<Bytes, Error> {
        Box::pin(async move {
            let shared_credit_bytes = match &self.response_body {
                ResponseBody::Shared(Some(body)) => body.reserved_bytes(),
                _ => 0,
            };
            let bytes = match self.response_body {
                ResponseBody::Shared(body) => {
                    body.map_or_else(Bytes::new, |body| body.into_bytes())
                }
                ResponseBody::Disk(body) => body.map_or_else(Bytes::new, |(bytes, _slot)| bytes),
                ResponseBody::Scheduled(Some(body)) => body.read().await?,
                ResponseBody::Scheduled(None) => Bytes::new(),
                ResponseBody::File(file) => file.map_err(|e| Error::FetchError(e.to_string()))?,
                ResponseBody::Network(response) => {
                    let response = response
                        .lock()
                        .expect("working lock during fetch body read")
                        .take()
                        .expect("Body cannot already be consumed");
                    response
                        .bytes()
                        .await
                        .map_err(|e| Error::FetchError(e.to_string()))?
                }
            };
            if let Some(request) = self.loader_request {
                tracing::debug!(
                    target: "ruffle_loader",
                    stage = "io",
                    session_generation = request.session_generation,
                    request_id = request.request_id,
                    resource_version = request.resource_version,
                    input_bytes = bytes.len(),
                    shared_credit_bytes,
                    "Loader I/O body delivered"
                );
            }
            Ok(bytes)
        })
    }

    fn text_encoding(&self) -> Option<&'static Encoding> {
        self.text_encoding
    }

    fn status(&self) -> u16 {
        self.status
    }

    fn redirected(&self) -> bool {
        self.redirected
    }

    #[expect(clippy::await_holding_lock)]
    fn next_chunk(&mut self) -> OwnedFuture<Option<Vec<u8>>, Error> {
        match &mut self.response_body {
            ResponseBody::Shared(data) => {
                let data = data.take();
                Box::pin(async move {
                    let bytes = data.map(|body| body.into_bytes());
                    Ok(bytes.filter(|b| !b.is_empty()).map(Into::into))
                })
            }
            ResponseBody::Disk(body) => {
                let body = body.take();
                Box::pin(async move { Ok(body.map(|(bytes, _slot)| bytes.into())) })
            }
            ResponseBody::Scheduled(body) => {
                let body = body.take();
                Box::pin(async move {
                    match body {
                        Some(body) => body.read().await.map(|bytes| Some(bytes.into())),
                        None => Ok(None),
                    }
                })
            }
            ResponseBody::File(file) => {
                let res = file
                    .as_mut()
                    .map(std::mem::take)
                    .map_err(|e| Error::FetchError(e.to_string()));

                Box::pin(async move {
                    match res {
                        Ok(bytes) if !bytes.is_empty() => Ok(Some(bytes.into())),
                        Ok(_) => Ok(None),
                        Err(e) => Err(e),
                    }
                })
            }
            ResponseBody::Network(response) => {
                let response = response.clone();
                Box::pin(async move {
                    let lock = response.try_lock();
                    if matches!(lock, Err(std::sync::TryLockError::WouldBlock)) {
                        return Err(Error::FetchError(
                            "Concurrent read operations on the same stream are not supported."
                                .to_string(),
                        ));
                    }

                    let result = lock
                        .expect("desktop network lock")
                        .as_mut()
                        .expect("Body cannot already be consumed")
                        .chunk()
                        .await;

                    match result {
                        Ok(Some(bytes)) => Ok(Some(bytes.to_vec())),
                        Ok(None) => Ok(None),
                        Err(e) => Err(Error::FetchError(e.to_string())),
                    }
                })
            }
        }
    }

    fn buffered_body_len(&self) -> Option<usize> {
        match &self.response_body {
            ResponseBody::Shared(body) => body.as_ref().map(|body| body.len()),
            ResponseBody::Disk(body) => body.as_ref().map(|(bytes, _)| bytes.len()),
            ResponseBody::File(Ok(bytes)) => Some(bytes.len()),
            _ => None,
        }
    }

    fn expected_length(&self) -> Result<Option<u64>, Error> {
        match &self.response_body {
            ResponseBody::Shared(body) => Ok(body.as_ref().map(|b| b.len() as u64)),
            ResponseBody::Disk(body) => Ok(body.as_ref().map(|(bytes, _)| bytes.len() as u64)),
            ResponseBody::Scheduled(body) => Ok(body.as_ref().and_then(|b| b.expected)),
            ResponseBody::File(file) => Ok(file.as_ref().map(|file| file.len() as u64).ok()),
            ResponseBody::Network(response) => {
                let lock = response.lock().expect("no recursive locks");
                let response = lock.as_ref().expect("Body cannot already be consumed");
                Ok(response.content_length())
            }
        }
    }
}
