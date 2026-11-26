//! Resource transport and URL policy shared by streaming and buffered Flash consumers.
use crate::backend::navigator::{
    ErrorResponse, FetchReason, NavigatorBackend, OwnedFuture, Request, SuccessResponse,
};
use crate::compatibility_rules::{CompatibilityRules, UrlRewriteStage};
use crate::context::UpdateContext;
use crate::loader::Error;
use crate::player::Player;
use futures::future::{AbortHandle, AbortRegistration, Abortable};
use slotmap::{DefaultKey, SlotMap};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Mutex, Weak};

/// A pending result may outlive either its owner read or the player.
pub(super) struct ReadTarget {
    player: Weak<Mutex<Player>>,
    abort: AbortHandle,
}

impl ReadTarget {
    pub fn new(context: &UpdateContext<'_>, registration: &AbortRegistration) -> Self {
        Self {
            player: context.player.clone(),
            abort: registration.handle(),
        }
    }

    pub fn fetch(
        &self,
        start: impl FnOnce(&Player) -> OwnedFuture<Box<dyn SuccessResponse>, ErrorResponse> + 'static,
    ) -> OwnedFuture<Box<dyn SuccessResponse>, ErrorResponse> {
        let weak = self.player.clone();
        Box::pin(async move {
            let fetch = weak.upgrade().map(|player| start(&player.lock().unwrap()));
            match fetch {
                Some(fetch) => fetch.await,
                None => Err(ErrorResponse {
                    url: String::new(),
                    error: Error::Cancelled,
                }),
            }
        })
    }

    pub fn cancelled(&self) -> bool {
        self.abort.is_aborted() || self.player.strong_count() == 0
    }

    pub fn deliver(
        &self,
        callback: impl FnOnce(&mut UpdateContext<'_>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let Some(player) = self.player.upgrade().filter(|_| !self.abort.is_aborted()) else {
            return Ok(());
        };
        player.lock().unwrap().update(|context| {
            if self.abort.is_aborted() {
                return Ok(());
            }
            callback(context)
        })
    }
}

enum BodyMode {
    Root,
    Buffered,
    Streaming,
}

pub(crate) struct Resources {
    rules: Rc<CompatibilityRules>,
    // AVM1 resources and NetStream have no common AVM2 Loader owner.
    reads: Rc<RefCell<SlotMap<DefaultKey, AbortHandle>>>,
}

impl Resources {
    pub fn new(rules: CompatibilityRules) -> Self {
        Self {
            rules: Rc::new(rules),
            reads: Default::default(),
        }
    }

    pub fn cancel_reads(&self) {
        for (_, read) in self.reads.borrow_mut().drain() {
            read.abort();
        }
    }

    pub(super) fn run_read(
        &self,
        registration: AbortRegistration,
        future: impl Future<Output = Result<(), Error>> + 'static,
    ) -> OwnedFuture<(), Error> {
        let id = self.reads.borrow_mut().insert(registration.handle());
        let reads = self.reads.clone();
        let entry = scopeguard::guard(id, move |id| {
            reads.borrow_mut().remove(id);
        });
        Box::pin(async move {
            let _entry = entry;
            Abortable::new(future, registration).await.unwrap_or(Ok(()))
        })
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

#[cfg(test)]
mod read_tests;

#[cfg(test)]
mod tests {
    use crate::tag_utils::SwfMovieData;
    use futures::{FutureExt, channel::oneshot};
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn replacing_the_root_cancels_old_data_reads_before_their_vm_callbacks() {
        let session = crate::PlayerBuilder::new().build_session();
        let player = session.player();
        let completed = Rc::new(Cell::new(0));
        let start = || {
            let (send, receive) = oneshot::channel::<()>();
            let completed = completed.clone();
            let task = player.lock().unwrap().mutate_with_update_context(|ctx| {
                let target = ctx.avm2.classes().object.into();
                ctx.load_manager.cancellable_read(
                    target,
                    player,
                    None,
                    Box::pin(async move {
                        receive.await.unwrap();
                        completed.set(completed.get() + 1);
                        Ok(())
                    }),
                )
            });
            (send, task)
        };
        let (late, mut old) = start();
        assert!(old.as_mut().now_or_never().is_none());
        let (before, after) = player.lock().unwrap().mutate_with_update_context(|ctx| {
            let before = ctx.load_manager.session_generation;
            ctx.replace_root_movie(SwfMovieData::empty(10, None));
            (before, ctx.load_manager.session_generation)
        });
        old.now_or_never()
            .expect("old resource still pending after root replacement")
            .unwrap();
        assert!(late.send(()).is_err());
        assert_ne!(before, after);
        let (deliver, current) = start();
        deliver.send(()).unwrap();
        futures::executor::block_on(current).unwrap();
        assert_eq!(completed.get(), 1);
    }
}
