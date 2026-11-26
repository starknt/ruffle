use parking_lot::{Condvar, Mutex};
use std::{
    io,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

// A short burst of pipeline creation (including the initial scene load) needs one export.
const BATCH_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Default, Debug)]
struct Pending {
    generation: u64,
    saved: u64,
    requested: u64,
    completed: u64,
    due: Option<Instant>,
    stopping: bool,
}

#[derive(Debug)]
pub(super) struct Writer {
    state: Arc<(Mutex<Pending>, Condvar)>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Writer {
    pub fn new(mut export: impl FnMut() -> bool + Send + 'static) -> io::Result<Self> {
        let state = Arc::new((Mutex::new(Pending::default()), Condvar::new()));
        let shared = state.clone();
        let worker = thread::Builder::new()
            .name("wgpu-cache-save".into())
            .spawn(move || {
                let (mutex, wake) = &*shared;
                let mut pending = mutex.lock();
                loop {
                    if pending.stopping && pending.due.is_none() {
                        break;
                    }
                    let Some(due) = pending.due else {
                        wake.wait(&mut pending);
                        continue;
                    };
                    if !pending.stopping && due > Instant::now() {
                        wake.wait_until(&mut pending, due);
                        continue;
                    }
                    pending.due = None;
                    let (generation, request) = (pending.generation, pending.requested);
                    if generation > pending.saved {
                        drop(pending);
                        let ok = export();
                        pending = mutex.lock();
                        if ok {
                            pending.saved = generation;
                        }
                    }
                    pending.completed = request;
                    wake.notify_all();
                }
            })?;
        Ok(Self {
            state,
            worker: Some(worker),
        })
    }

    pub fn dirty(&self) {
        let (mutex, wake) = &*self.state;
        let mut p = mutex.lock();
        p.generation += 1;
        if p.due.is_none() {
            p.due = Some(Instant::now() + BATCH_INTERVAL);
            wake.notify_one();
        }
    }

    pub fn save(&self, wait: bool) -> bool {
        let (mutex, wake) = &*self.state;
        let mut p = mutex.lock();
        let target = p.generation;
        if p.saved == target {
            return target != 0;
        }
        p.due = Some(Instant::now());
        p.requested += 1;
        let request = p.requested;
        wake.notify_all();
        while wait && p.saved < target && p.completed < request {
            wake.wait(&mut p);
        }
        !wait || p.saved >= target
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        self.state.0.lock().stopping = true;
        self.state.1.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    #[test]
    fn creation_during_export_and_failed_exports_can_be_saved_again() {
        let (started, exports) = mpsc::channel();
        let (finish, next) = mpsc::channel();
        let writer = Writer::new(move || {
            started.send(()).unwrap();
            next.recv().unwrap()
        })
        .unwrap();
        assert!(!writer.save(true));
        writer.dirty();
        writer.dirty();
        writer.save(false);
        exports.recv_timeout(Duration::from_secs(3)).unwrap();
        writer.dirty();
        finish.send(true).unwrap();
        std::thread::scope(|scope| {
            let failed = scope.spawn(|| writer.save(true));
            exports.recv_timeout(Duration::from_secs(3)).unwrap();
            finish.send(false).unwrap();
            assert!(!failed.join().unwrap());
            let retry = scope.spawn(|| writer.save(true));
            exports.recv_timeout(Duration::from_secs(3)).unwrap();
            finish.send(true).unwrap();
            assert!(retry.join().unwrap());
        });
        assert!(writer.save(true));
        assert!(exports.try_recv().is_err());
    }
}
