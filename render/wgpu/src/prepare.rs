//! Native CPU preparation on the shared Rayon facility.
pub mod mesh;

use ruffle_render::error::{RenderFailure, RenderFailureKind, RendererHealth};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

pub struct Work<T> {
    pub value: T,
    pub bytes: u64,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct PreparationStats {
    pub active: usize,
    pub bytes: u64,
    pub peak_active: usize,
    pub peak_bytes: u64,
    pub submitted: u64,
}
struct Shared {
    limit: usize,
    scratch_limit: u64,
    stats: Mutex<PreparationStats>,
    health: RendererHealth,
}
#[derive(Clone)]
pub struct NativePreparation(Arc<Shared>);
struct Permit(Arc<Shared>, u64);
impl Drop for Permit {
    fn drop(&mut self) {
        let mut stats = self
            .0
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        stats.active -= 1;
        stats.bytes -= self.1;
    }
}
#[derive(Clone)]
pub struct PreparationGeneration(Arc<AtomicU64>);
impl Default for PreparationGeneration {
    fn default() -> Self {
        Self(Arc::new(AtomicU64::new(1)))
    }
}
#[derive(Clone)]
pub struct GenerationToken {
    value: u64,
    current: Arc<AtomicU64>,
}
impl GenerationToken {
    fn is_current(&self) -> bool {
        self.value != 0 && self.current.load(Ordering::Acquire) == self.value
    }
}
impl PreparationGeneration {
    pub fn capture(&self) -> GenerationToken {
        GenerationToken {
            value: self.0.load(Ordering::Acquire),
            current: self.0.clone(),
        }
    }
    pub fn invalidate(&self) {
        if self
            .0
            .try_update(Ordering::AcqRel, Ordering::Acquire, |v| {
                v.checked_add(1).filter(|_| v != 0)
            })
            .is_err()
        {
            self.cancel();
        }
    }
    pub fn cancel(&self) {
        self.0.store(0, Ordering::Release);
    }
}
pub fn parallel_workload(commands: usize, vertices: usize) -> bool {
    commands >= 256 || vertices >= 8192
}
impl NativePreparation {
    pub fn new(limit: usize, scratch_limit: u64, health: RendererHealth) -> Self {
        Self(Arc::new(Shared {
            limit: limit.clamp(1, 4),
            scratch_limit,
            stats: Mutex::default(),
            health,
        }))
    }
    pub fn for_device(scratch_limit: u64, health: RendererHealth) -> Self {
        let cpus = std::thread::available_parallelism().map_or(1, usize::from);
        Self::new(
            cpus.saturating_sub(2).clamp(1, 4),
            scratch_limit.min(64 * 1024 * 1024),
            health,
        )
    }
    pub fn stats(&self) -> PreparationStats {
        *self
            .0
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    fn admit(&self, bytes: u64, dispatched: bool) -> Option<Permit> {
        let mut stats = self
            .0
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if stats.active >= self.0.limit || bytes > self.0.scratch_limit.saturating_sub(stats.bytes)
        {
            return None;
        }
        stats.active += 1;
        stats.bytes += bytes;
        stats.peak_active = stats.peak_active.max(stats.active);
        stats.peak_bytes = stats.peak_bytes.max(stats.bytes);
        stats.submitted += u64::from(dispatched);
        Some(Permit(self.0.clone(), bytes))
    }
    fn admit_serial(&self, bytes: u64) -> Result<Permit, RenderFailure> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            self.0.health.check()?;
            if let Some(permit) = self.admit(bytes.min(self.0.scratch_limit), false) {
                return Ok(permit);
            }
            if std::time::Instant::now() >= deadline {
                return Err(RenderFailure::new(
                    RenderFailureKind::Timeout,
                    "native preparation scratch remains in use",
                ));
            }
            // Another renderer can own the shared pool. Never wait with its
            // accounting lock held, or overlap its retained output uncharged.
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    /// Large mandatory shapes may borrow the whole preparation allowance, after
    /// all regular waves have been consumed. Keep the exclusive reservation
    /// through owner-thread publication, just like a normal worker result.
    pub fn run_exclusive<R>(
        &self,
        generation: &GenerationToken,
        prepare: impl FnOnce(u64) -> Result<R, RenderFailure>,
        consume: impl FnOnce(R) -> Result<(), RenderFailure>,
    ) -> Result<bool, RenderFailure> {
        let _permit = self.admit_serial(self.0.scratch_limit)?;
        if !generation.is_current() {
            return Ok(false);
        }
        let timer =
            ruffle_render::profile::native_time(ruffle_render::profile::NativePhase::Preparation);
        let output = prepare(self.0.scratch_limit)?;
        drop(timer);
        self.0.health.check()?;
        if !generation.is_current() {
            return Ok(false);
        }
        consume(output)?;
        Ok(true)
    }
    /// Join each bounded wave, then consume it on the caller thread before
    /// admitting another wave. GC/GPU ownership never enters the Rayon closure.
    pub fn for_each_batch<T: Send, R: Send>(
        &self,
        generation: &GenerationToken,
        parallel: bool,
        work: impl IntoIterator<Item = Work<T>>,
        prepare: impl Fn(T) -> Result<R, RenderFailure> + Send + Sync,
        mut consume: impl FnMut(Vec<R>) -> Result<(), RenderFailure>,
    ) -> Result<bool, RenderFailure> {
        let mut work = work.into_iter().peekable();
        loop {
            self.0.health.check()?;
            if !generation.is_current() {
                return Ok(false);
            }
            let timer = ruffle_render::profile::native_time(
                ruffle_render::profile::NativePhase::Preparation,
            );
            let mut pending = Vec::new();
            let mut permits = Vec::new();
            while parallel && pending.len() < self.0.limit {
                let Some(next) = work.peek() else {
                    break;
                };
                let Some(permit) = self.admit(next.bytes, true) else {
                    break;
                };
                permits.push(permit);
                pending.push(work.next().expect("peeked preparation work").value);
            }
            if pending.is_empty() {
                let Some(item) = work.next() else {
                    return Ok(true);
                };
                // A serial fallback is consumed alone. In particular, multiple
                // oversized outputs must never accumulate in a parallel wave.
                let _permit = self.admit_serial(item.bytes)?;
                if !generation.is_current() {
                    return Ok(false);
                }
                let result = prepare(item.value)?;
                self.0.health.check()?;
                if !generation.is_current() {
                    return Ok(false);
                }
                drop(timer);
                consume(vec![result])?;
                continue;
            }
            let results: Vec<Mutex<Option<Result<R, RenderFailure>>>> =
                (0..pending.len()).map(|_| Mutex::new(None)).collect();
            rayon::scope(|scope| {
                for (index, value) in pending.into_iter().enumerate() {
                    let result = &results[index];
                    let prepare = &prepare;
                    scope.spawn(move |_| {
                        if generation.is_current() {
                            *result
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                Some(prepare(value));
                        }
                    });
                }
            });
            self.0.health.check()?;
            if !generation.is_current() {
                return Ok(false);
            }
            let output = results
                .into_iter()
                .map(|result| {
                    result
                        .into_inner()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .ok_or_else(|| {
                            RenderFailure::new(
                                RenderFailureKind::Internal,
                                "native preparation returned no result",
                            )
                        })?
                })
                .collect::<Result<Vec<_>, _>>()?;
            drop(timer);
            consume(output)?;
            // Reservations include completed, unpublished output, not just the
            // worker's running interval. Another renderer shares these permits.
            drop(permits);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruffle_render::error::{RenderFailure, RenderFailureKind, RendererHealth};
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn native_preparation_keeps_order_and_bounds_simultaneous_scratch() {
        let pool = NativePreparation::new(2, 128, RendererHealth::default());
        let generation = PreparationGeneration::default();
        let mut output = Vec::new();
        assert!(
            pool.for_each_batch(
                &generation.capture(),
                true,
                (0..21).map(|n| Work {
                    value: n,
                    bytes: 64
                }),
                |n| Ok(n * n + 1),
                |batch| {
                    output.extend(batch);
                    Ok(())
                }
            )
            .unwrap()
        );
        assert_eq!(
            output,
            [
                1, 2, 5, 10, 17, 26, 37, 50, 65, 82, 101, 122, 145, 170, 197, 226, 257, 290, 325,
                362, 401
            ]
        );
        let stats = pool.stats();
        assert!(stats.submitted > 0);
        assert!(stats.peak_active <= 2 && stats.peak_bytes <= 128);
        assert_eq!((stats.active, stats.bytes), (0, 0));
    }

    #[test]
    fn native_preparation_does_not_publish_stale_generation_or_failed_results() {
        let health = RendererHealth::default();
        let pool = NativePreparation::new(2, 128, health.clone());
        let generation = PreparationGeneration::default();
        let captured = generation.capture();
        let barrier = Arc::new(Barrier::new(2));
        let published = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let barrier2 = barrier.clone();
            let generation = &generation;
            scope.spawn(move || {
                barrier2.wait();
                generation.invalidate();
                barrier2.wait();
            });
            let current = pool
                .for_each_batch(
                    &captured,
                    true,
                    [Work {
                        value: (),
                        bytes: 64,
                    }],
                    |_| {
                        barrier.wait();
                        barrier.wait();
                        Ok(42)
                    },
                    |_| {
                        published.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    },
                )
                .unwrap();
            assert!(!current);
        });
        assert_eq!(published.load(Ordering::SeqCst), 0);
        let error = pool
            .for_each_batch(
                &generation.capture(),
                true,
                [Work {
                    value: (),
                    bytes: 64,
                }],
                |_| {
                    health.record(RenderFailure::new(
                        RenderFailureKind::DeviceLost,
                        "injected device loss during CPU work",
                    ));
                    Ok(42)
                },
                |_| {
                    published.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
            )
            .unwrap_err();
        assert_eq!(error.kind(), RenderFailureKind::DeviceLost);
        assert_eq!(published.load(Ordering::SeqCst), 0);
        assert_eq!((pool.stats().active, pool.stats().bytes), (0, 0));
    }

    #[test]
    fn native_preparation_small_or_oversized_inputs_stay_on_the_owner_thread() {
        let pool = NativePreparation::new(2, 128, RendererHealth::default());
        let generation = PreparationGeneration::default();
        let owner = std::thread::current().id();
        for (parallel, bytes) in [(false, 64), (true, 129)] {
            pool.for_each_batch(
                &generation.capture(),
                parallel,
                [Work { value: 3, bytes }],
                |n| {
                    assert_eq!(std::thread::current().id(), owner);
                    Ok(n)
                },
                |batch| {
                    assert_eq!(batch, [3]);
                    Ok(())
                },
            )
            .unwrap();
        }
        assert_eq!(pool.stats().submitted, 0);
        assert!(!parallel_workload(255, 8191));
        assert!(parallel_workload(256, 0));
        assert!(parallel_workload(0, 8192));
    }
}

#[cfg(test)]
#[test]
fn native_preparation_accounts_outputs_through_publication_and_consumes_oversized_work_alone() {
    let pool = NativePreparation::new(4, 128, RendererHealth::default());
    let generation = PreparationGeneration::default();
    pool.for_each_batch(
        &generation.capture(),
        true,
        (0..6).map(|n| Work {
            value: n,
            bytes: 64,
        }),
        Ok,
        |batch| {
            assert_eq!(pool.stats().bytes, batch.len() as u64 * 64);
            assert!(batch.len() <= 2);
            Ok(())
        },
    )
    .unwrap();
    pool.for_each_batch(
        &generation.capture(),
        true,
        (0..6).map(|n| Work {
            value: n,
            bytes: 256,
        }),
        Ok,
        |batch| {
            assert_eq!(batch.len(), 1);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(pool.stats().bytes, 0);
}

#[cfg(test)]
#[test]
fn native_preparation_serial_reserves_scratch_through_publication() {
    let pool = NativePreparation::new(2, 128, RendererHealth::default());
    pool.for_each_batch(
        &PreparationGeneration::default().capture(),
        false,
        [Work {
            value: 42,
            bytes: 64,
        }],
        |value| {
            assert_eq!(pool.stats().bytes, 64);
            Ok(value)
        },
        |batch| {
            assert_eq!(batch, [42]);
            assert_eq!(pool.stats().bytes, 64);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!((pool.stats().active, pool.stats().bytes), (0, 0));
}

#[cfg(test)]
#[test]
fn native_preparation_exclusive_output_stays_reserved_and_rejects_stale_publication() {
    let pool = NativePreparation::new(2, 128, RendererHealth::default());
    let generation = PreparationGeneration::default();
    assert!(
        pool.run_exclusive(
            &generation.capture(),
            |budget| {
                assert_eq!(budget, 128);
                assert_eq!(pool.stats().bytes, 128);
                Ok(42)
            },
            |value| {
                assert_eq!(value, 42);
                assert_eq!(pool.stats().bytes, 128);
                Ok(())
            },
        )
        .unwrap()
    );
    assert!(
        !pool
            .run_exclusive(
                &generation.capture(),
                |_| {
                    generation.invalidate();
                    Ok(42)
                },
                |_| panic!("stale mesh must not publish"),
            )
            .unwrap()
    );
    assert_eq!((pool.stats().active, pool.stats().bytes), (0, 0));
}
