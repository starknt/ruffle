//! Resource transport and URL policy shared by streaming and buffered Flash consumers.
use crate::backend::navigator::{
    ErrorResponse, FetchReason, NavigatorBackend, OwnedFuture, Request, SuccessResponse,
};
use crate::compatibility_rules::{CompatibilityRules, UrlRewriteStage};
use std::rc::Rc;

enum BodyMode {
    Root,
    Buffered,
    Streaming,
}

pub(crate) struct Resources {
    rules: Rc<CompatibilityRules>,
}

impl Resources {
    pub fn new(rules: CompatibilityRules) -> Self {
        Self {
            rules: Rc::new(rules),
        }
    }

    pub fn rules(&self) -> &CompatibilityRules {
        &self.rules
    }

    pub fn fetch(
        &self,
        navigator: &dyn NavigatorBackend,
        request: Request,
        reason: FetchReason,
    ) -> OwnedFuture<Box<dyn SuccessResponse>, ErrorResponse> {
        self.request(
            navigator,
            request,
            reason,
            if matches!(reason, FetchReason::LoadSwf | FetchReason::UrlLoader) {
                BodyMode::Buffered
            } else {
                BodyMode::Streaming
            },
        )
    }

    pub fn root_movie(
        &self,
        navigator: &dyn NavigatorBackend,
        request: Request,
    ) -> OwnedFuture<Box<dyn SuccessResponse>, ErrorResponse> {
        self.request(navigator, request, FetchReason::LoadSwf, BodyMode::Root)
    }

    pub fn buffered_sound(
        &self,
        navigator: &dyn NavigatorBackend,
        request: Request,
    ) -> OwnedFuture<Box<dyn SuccessResponse>, ErrorResponse> {
        self.request(navigator, request, FetchReason::Other, BodyMode::Buffered)
    }

    fn request(
        &self,
        navigator: &dyn NavigatorBackend,
        mut request: Request,
        reason: FetchReason,
        mode: BodyMode,
    ) -> OwnedFuture<Box<dyn SuccessResponse>, ErrorResponse> {
        match self.rules.block_or_rewrite_swf_url(
            request.url().into(),
            UrlRewriteStage::BeforeRequest,
            reason,
        ) {
            Ok(Some(url)) => request.set_url(url),
            Ok(None) => {}
            Err(error) => return Box::pin(async move { Err(error) }),
        }
        let fetch = match mode {
            BodyMode::Root => navigator.fetch_root(request),
            BodyMode::Buffered => navigator.fetch_buffered(request, reason),
            BodyMode::Streaming => navigator.fetch(request),
        };
        let rules = self.rules.clone();
        Box::pin(async move {
            let mut response = fetch.await?;
            if let Some(url) = rules.block_or_rewrite_swf_url(
                response.url(),
                UrlRewriteStage::AfterResponse,
                reason,
            )? {
                response.set_url(url);
            }
            Ok(response)
        })
    }
}
