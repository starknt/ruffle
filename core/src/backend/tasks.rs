//! Player-thread futures may capture AVM handles. Hosts drive them outside the player lock.
use crate::backend::navigator::OwnedFuture;
use crate::loader::Error;
use futures::executor::{LocalPool, LocalSpawner};
use futures::task::LocalSpawnExt;
pub use futures::task::SpawnError;
use std::cell::Cell;
use std::rc::Rc;

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
}
