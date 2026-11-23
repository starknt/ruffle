//! Player-thread futures may capture AVM handles. Hosts drive them outside the player lock.
use crate::loader::Error;
use futures::executor::{LocalPool, LocalSpawner};
pub use futures::task::SpawnError;
use futures::task::{AtomicWaker, LocalSpawn, LocalSpawnExt};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Wake, Waker};

pub type OwnedFuture<T, E> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, E>> + 'static>>;

pub type PlayerTask = OwnedFuture<(), Error>;

pub trait TaskSpawner {
    /// Accept a task without polling it inline. A closed session rejects new tasks.
    fn try_spawn(&self, task: PlayerTask) -> Result<(), SpawnError>;

    fn spawn(&self, task: PlayerTask) {
        if let Err(error) = self.try_spawn(task) {
            tracing::error!("Could not schedule player task: {error}");
        }
    }
}

#[derive(Default)]
pub struct LocalExecutor {
    pool: LocalPool,
    pending: Rc<Cell<usize>>,
}

impl LocalExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spawner(&self) -> LocalTaskSpawner {
        LocalTaskSpawner {
            spawner: self.pool.spawner(),
            pending: self.pending.clone(),
        }
    }

    pub fn pending_tasks(&self) -> usize {
        self.pending.get()
    }

    pub fn run(&mut self) {
        self.pool.run_until_stalled();
    }
}

#[derive(Clone)]
pub struct LocalTaskSpawner {
    spawner: LocalSpawner,
    pending: Rc<Cell<usize>>,
}

/// Cross-thread callbacks only wake the local queue and the host. The player thread
/// owns the future and drops it when the session closes, even if it never wakes.
struct HostWake<F> {
    local: AtomicWaker,
    notify: F,
}

impl<F: Fn() -> Result<(), SpawnError> + Send + Sync + 'static> Wake for HostWake<F> {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        if let Some(local) = self.local.take() {
            local.wake();
            let _ = (self.notify)();
        }
    }
}

impl LocalTaskSpawner {
    /// Queue a task for a host event loop. The callback must only post an event;
    /// the host calls `LocalExecutor::run` on the player thread when it receives it.
    pub fn try_spawn_with_wake(
        &self,
        mut task: PlayerTask,
        notify: impl Fn() -> Result<(), SpawnError> + Send + Sync + 'static,
    ) -> Result<(), SpawnError> {
        self.spawner.status_local()?;
        notify()?;
        let wake = Arc::new(HostWake {
            local: AtomicWaker::new(),
            notify,
        });
        let waker = Waker::from(wake.clone());
        self.try_spawn(Box::pin(futures::future::poll_fn(move |cx| {
            wake.local.register(cx.waker());
            task.as_mut().poll(&mut Context::from_waker(&waker))
        })))
    }
}

struct PendingTask(Rc<Cell<usize>>);

impl Drop for PendingTask {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

impl TaskSpawner for LocalTaskSpawner {
    fn try_spawn(&self, task: PlayerTask) -> Result<(), SpawnError> {
        self.pending.set(self.pending.get() + 1);
        let pending = PendingTask(self.pending.clone());
        self.spawner.spawn_local(async move {
            let _pending = pending;
            if let Err(error) = task.await {
                tracing::error!("Asynchronous player error: {error}");
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_event_loop_wakes_for_io_and_closing_drops_a_sleeping_task() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let mut executor = LocalExecutor::new();
        let spawner = executor.spawner();
        let wakes = Arc::new(AtomicUsize::new(0));
        let (ready, wait) = futures::channel::oneshot::channel::<()>();
        let held = Rc::new(());
        let task_held = held.clone();
        let notified = wakes.clone();
        spawner
            .try_spawn_with_wake(
                Box::pin(async move {
                    let _held = task_held;
                    wait.await.unwrap();
                    futures::future::pending().await
                }),
                move || {
                    notified.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(wakes.load(Ordering::Relaxed), 1);
        executor.run();
        std::thread::spawn(move || ready.send(()).unwrap())
            .join()
            .unwrap();
        assert_eq!(wakes.load(Ordering::Relaxed), 2);
        executor.run();
        assert_eq!(Rc::strong_count(&held), 2);
        drop(executor);
        assert_eq!(Rc::strong_count(&held), 1);
        assert_eq!(spawner.pending.get(), 0);
    }

    #[test]
    fn a_closed_event_loop_rejects_a_task_before_it_enters_the_local_queue() {
        let executor = LocalExecutor::new();
        let spawner = executor.spawner();
        let error = spawner
            .try_spawn_with_wake(Box::pin(std::future::pending()), || {
                Err(SpawnError::shutdown())
            })
            .unwrap_err();
        assert!(error.is_shutdown());
        assert_eq!(executor.pending_tasks(), 0);
    }

    #[test]
    fn closing_executor_cancels_unpolled_tasks_and_rejects_new_ones() {
        let executor = LocalExecutor::new();
        let spawner = executor.spawner();
        spawner.try_spawn(Box::pin(std::future::pending())).unwrap();
        assert_eq!(executor.pending_tasks(), 1);
        drop(executor);
        assert_eq!(spawner.pending.get(), 0);
        assert!(
            spawner
                .try_spawn(Box::pin(async { Ok(()) }))
                .unwrap_err()
                .is_shutdown()
        );
        assert_eq!(spawner.pending.get(), 0);
    }
    #[test]
    fn local_executor_reports_pending_tasks_until_completion() {
        let mut executor = LocalExecutor::new();
        let spawner = executor.spawner();
        let (sender, receiver) = futures::channel::oneshot::channel::<()>();

        spawner.spawn(Box::pin(async move {
            let _ = receiver.await;
            Ok::<(), Error>(())
        }));

        assert_eq!(executor.pending_tasks(), 1);
        executor.run();
        assert_eq!(executor.pending_tasks(), 1);

        sender.send(()).unwrap();
        executor.run();
        assert_eq!(executor.pending_tasks(), 0);
    }
}
