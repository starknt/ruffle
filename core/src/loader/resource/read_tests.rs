use super::ReadTarget;
use crate::streams::NetStream;
use crate::tag_utils::SwfMovieData;
use futures::{FutureExt, channel::oneshot, future::AbortHandle};
use std::cell::Cell;
use std::rc::Rc;

#[test]
fn retired_reads_drop_held_results_while_independent_reads_finish() {
    let session = crate::PlayerBuilder::new().build_session();
    let delivered = Rc::new(Cell::new(0));
    let start = || {
        let (send, receive) = oneshot::channel::<()>();
        let (abort, registration) = AbortHandle::new_pair();
        let delivered = delivered.clone();
        let held_body = Rc::new([0u8; 256]);
        let held = Rc::downgrade(&held_body);
        let task = session.lock().unwrap().mutate_with_update_context(|uc| {
            let target = ReadTarget::new(uc, &registration);
            uc.resources.run_read(registration, async move {
                receive.await.unwrap();
                target.deliver(|_| {
                    delivered.set(delivered.get() + held_body.len());
                    Ok(())
                })
            })
        });
        (task, abort, send, held)
    };

    let (mut old, close, late, old_body) = start();
    assert!(old.as_mut().now_or_never().is_none());
    let (current, _, ready, _) = start();
    ready.send(()).unwrap();
    current.now_or_never().unwrap().unwrap();
    assert_eq!(delivered.get(), 256);

    close.abort();
    old.now_or_never().unwrap().unwrap();
    assert!(old_body.upgrade().is_none());
    assert!(late.send(()).is_err());

    let (mut pending, _, late, body) = start();
    assert!(pending.as_mut().now_or_never().is_none());
    session.lock().unwrap().mutate_with_update_context(|uc| {
        uc.replace_root_movie(SwfMovieData::empty(10, None));
    });
    pending.now_or_never().unwrap().unwrap();
    assert!(body.upgrade().is_none());
    assert!(late.send(()).is_err());
    assert_eq!(delivered.get(), 256);
}

#[test]
fn replacing_the_root_retires_buffered_stream_playback() {
    let session = crate::PlayerBuilder::new().build_session();
    session.lock().unwrap().mutate_with_update_context(|uc| {
        let stream = NetStream::new_avm2(uc.gc());
        stream.play(uc, None);
        stream.load_buffer(uc, &mut vec![1, 2, 3]);
        stream.finish_buffer();
        uc.replace_root_movie(SwfMovieData::empty(10, None));
        assert!(!uc.stream_manager.is_active());
        assert_eq!(stream.bytes_loaded(), 0);
    });
}
