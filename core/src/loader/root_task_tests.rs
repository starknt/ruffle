use super::run_file_dialog;
use crate::{PlayerBuilder, PlayerSession, loader::Error, tag_utils::SwfMovieData};
use futures::channel::oneshot;
use std::{cell::RefCell, rc::Rc, sync::Arc};

// Use the FileReference dialog path so a stalled task really holds a player and a callback.
fn queue_dialog(
    session: &PlayerSession,
    callbacks: &Rc<RefCell<Vec<&'static str>>>,
    name: &'static str,
) -> oneshot::Sender<()> {
    let (send, receive) = oneshot::channel();
    let callbacks = callbacks.clone();
    session.lock().unwrap().mutate_with_update_context(|uc| {
        let task = run_file_dialog(
            uc,
            Box::pin(async move { receive.await.map_err(|_| Error::Cancelled) }),
            move |_, result| {
                result?;
                callbacks.borrow_mut().push(name);
                Ok(())
            },
        );
        uc.spawn_future(task);
    });
    send
}

#[test]
fn root_replacement_discards_sleeping_and_ready_dialogs_and_runs_current_tasks() {
    let mut session = PlayerBuilder::new()
        .with_movie(SwfMovieData::empty(8, None))
        .build_session();
    let owners = Arc::strong_count(session.player());
    let callbacks = Rc::new(RefCell::new(Vec::new()));
    let ordinary = queue_dialog(&session, &callbacks, "ordinary");
    let sleeping = queue_dialog(&session, &callbacks, "sleeping");
    let ready = queue_dialog(&session, &callbacks, "ready");
    session.poll_tasks();
    ordinary.send(()).unwrap();
    session.poll_tasks();
    assert_eq!(&*callbacks.borrow(), &["ordinary"]);

    ready.send(()).unwrap();
    session.lock().unwrap().mutate_with_update_context(|uc| {
        uc.replace_root_movie(SwfMovieData::empty(9, None));
    });
    let current = queue_dialog(&session, &callbacks, "current");
    current.send(()).unwrap();
    session.poll_tasks();
    assert_eq!(&*callbacks.borrow(), &["ordinary", "current"]);
    assert!(sleeping.send(()).is_err());
    assert_eq!(session.pending_tasks(), 0);
    assert_eq!(Arc::strong_count(session.player()), owners);
}
