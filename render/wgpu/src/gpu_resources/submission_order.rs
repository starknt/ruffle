//! A reserved native stage must reach the driver before later reliable writes.
//! The acknowledgement ends at queue submission. GPU completion has its own fence.
use super::{CommandBuffer, Queue, RenderFailure, RenderFailureKind};
use crossbeam_channel::{Receiver, Sender};
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
pub(super) struct SubmissionOrder {
    pending: AtomicBool,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    issued: u64,
    turns: VecDeque<(u64, Receiver<()>)>,
}

/// Reserves queue order while the native executor finishes the encoder. Dropping
/// an unsubmitted reservation releases just that turn, including during unwinding.
pub struct ReservedSubmission {
    queue: Queue,
    id: u64,
    _completion: Sender<()>,
}

impl SubmissionOrder {
    pub(super) fn reserve(&self, queue: &Queue) -> Result<ReservedSubmission, RenderFailure> {
        let runtime = queue.0.runtime();
        runtime.check_health()?;
        let mut state = self.state.lock().unwrap();
        if state.turns.len() >= runtime.config().resource_limit {
            return Err(RenderFailure::new(
                RenderFailureKind::BudgetExceeded,
                "stage submission queue is full",
            ));
        }
        let id = state.issued.checked_add(1).ok_or_else(|| {
            RenderFailure::new(
                RenderFailureKind::Internal,
                "stage submission sequence exhausted",
            )
        })?;
        let (signal, done) = crossbeam_channel::bounded(0);
        state.issued = id;
        state.turns.push_back((id, done));
        self.pending.store(true, Ordering::Release);
        Ok(ReservedSubmission {
            queue: queue.clone(),
            id,
            _completion: signal,
        })
    }

    pub(super) fn wait_pending(&self, queue: &Queue) -> Result<(), RenderFailure> {
        if self.pending.load(Ordering::Acquire) {
            let through = self.state.lock().unwrap().issued;
            self.wait_through(queue, through)?;
        }
        Ok(())
    }

    fn wait_through(&self, queue: &Queue, through: u64) -> Result<(), RenderFailure> {
        let health = queue.0.runtime().health();
        loop {
            health.check()?;
            let done = {
                let state = self.state.lock().unwrap();
                state
                    .turns
                    .front()
                    .filter(|(id, _)| *id <= through)
                    .map(|(_, r)| r.clone())
            };
            let Some(done) = done else { return Ok(()) };
            let failed = health.failure_signal();
            crossbeam_channel::select_biased! {
                recv(failed) -> _ => return Err(health.failure().expect("failed device signal")),
                recv(done) -> _ => (),
            }
        }
    }
}

impl ReservedSubmission {
    pub fn submit(
        self,
        commands: impl IntoIterator<Item = CommandBuffer>,
    ) -> Result<wgpu::SubmissionIndex, RenderFailure> {
        self.queue.order().wait_through(&self.queue, self.id - 1)?;
        self.queue.submit_ordered(commands)
    }
}

impl Drop for ReservedSubmission {
    fn drop(&mut self) {
        let order = self.queue.order();
        let mut state = order.state.lock().unwrap();
        state.turns.retain(|(id, _)| *id != self.id);
        order
            .pending
            .store(!state.turns.is_empty(), Ordering::Release);
    }
}
