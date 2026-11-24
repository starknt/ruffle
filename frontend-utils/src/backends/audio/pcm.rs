//! The device consumes prepared PCM. Only the producer enters the mixer or its decoders.
use rtrb::{Consumer, RingBuffer};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::{thread, time::Duration};

struct Clock {
    playing: bool,
    simulated: bool,
    stopped: bool,
    served: u64,
    produced: u64,
}

struct Shared {
    clock: Mutex<Clock>,
    active: AtomicBool,
    requested: AtomicU64,
    period: AtomicUsize,
    discard_before: AtomicU64,
}

pub(super) struct PcmWorker {
    shared: Arc<Shared>,
    thread: Option<thread::JoinHandle<()>>,
}

pub(super) struct PcmOutput<T> {
    shared: Arc<Shared>,
    samples: Consumer<T>,
    channels: usize,
    consumed: u64,
    silence: T,
}

impl<T: Copy> PcmOutput<T> {
    pub(super) fn fill(&mut self, buffer: &mut [T]) {
        if !self.shared.active.load(Ordering::Acquire) {
            buffer.fill(self.silence);
            return;
        }
        let floor = self.shared.discard_before.load(Ordering::Acquire);
        if floor > self.consumed {
            // Only the consumer changes the read cursor. The producer published
            // these samples before it published the new clock boundary.
            let count = (floor - self.consumed) as usize;
            self.samples
                .read_chunk(count)
                .expect("published PCM")
                .commit_all();
            self.consumed = floor;
        }
        let len = buffer.len() / self.channels * self.channels;
        let (frames, trailing) = buffer.split_at_mut(len);
        let (played, underrun) = self.samples.pop_partial_slice(frames);
        self.consumed += played.len() as u64;
        underrun.fill(self.silence);
        trailing.fill(self.silence);
        self.shared.period.store(len, Ordering::Relaxed);
        self.shared
            .requested
            .fetch_add(len as u64, Ordering::Release);
    }

    #[cfg(test)]
    pub(super) fn available(&self) -> usize {
        self.samples.slots()
    }
}

impl PcmWorker {
    pub(super) fn start<T: Copy + Send + 'static>(
        channels: usize,
        rate: u32,
        silence: T,
        mut mix: impl FnMut(&mut [T]) + Send + 'static,
    ) -> std::io::Result<(Self, PcmOutput<T>)> {
        if !(1..=255).contains(&channels) || rate == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid PCM format",
            ));
        }
        // Adapt the working set to the actual callback length. The ring and
        // scratch block are allocated once, even if a driver changes periods.
        let capacity = (rate as usize / 4).max(4096) * channels;
        let (mut writer, samples) = RingBuffer::new(capacity);
        let shared = Arc::new(Shared {
            clock: Mutex::new(Clock {
                playing: true,
                simulated: false,
                stopped: false,
                served: 0,
                produced: 0,
            }),
            active: AtomicBool::new(true),
            requested: AtomicU64::new(0),
            period: AtomicUsize::new(0),
            discard_before: AtomicU64::new(0),
        });
        let producer = shared.clone();
        let thread = thread::Builder::new()
            .name("ruffle-pcm".into())
            .spawn(move || {
                let mut block = vec![silence; 256 * channels];
                loop {
                    let mut clock = producer.clock.lock().unwrap();
                    if clock.stopped {
                        break;
                    }
                    if !clock.playing || clock.simulated {
                        drop(clock);
                        thread::park();
                        continue;
                    }
                    let requested = producer.requested.load(Ordering::Acquire);
                    let period = producer.period.load(Ordering::Relaxed).min(capacity);
                    let queued = capacity - writer.slots();
                    let pending =
                        requested.saturating_sub(clock.served).min(period as u64) as usize;
                    let count = pending.min(period.saturating_sub(queued)).min(block.len());
                    if count == 0 {
                        drop(clock);
                        let frames = period / channels;
                        let micros = (frames as u64 * 250_000 / u64::from(rate)).clamp(500, 4000);
                        thread::park_timeout(Duration::from_micros(micros));
                        continue;
                    }
                    mix(&mut block[..count]);
                    writer
                        .push_entire_slice(&block[..count])
                        .expect("reserved PCM space");
                    clock.produced += count as u64;
                    clock.served = requested - (pending - count) as u64;
                }
            })?;
        let output = PcmOutput {
            shared: shared.clone(),
            samples,
            channels,
            consumed: 0,
            silence,
        };
        Ok((
            Self {
                shared,
                thread: Some(thread),
            },
            output,
        ))
    }

    pub(super) fn set_playing(&mut self, playing: bool) {
        let mut clock = self.shared.clock.lock().unwrap();
        clock.playing = playing;
        self.shared
            .active
            .store(playing && !clock.simulated, Ordering::Release);
        self.thread.as_ref().unwrap().thread().unpark();
    }

    pub(super) fn set_simulated(&mut self, simulated: bool) {
        let mut clock = self.shared.clock.lock().unwrap();
        if clock.simulated != simulated {
            clock.simulated = simulated;
            clock.served = self.shared.requested.load(Ordering::Acquire);
            self.shared
                .discard_before
                .store(clock.produced, Ordering::Release);
            self.shared
                .active
                .store(clock.playing && !simulated, Ordering::Release);
            self.thread.as_ref().unwrap().thread().unpark();
        }
    }
}

impl Drop for PcmWorker {
    fn drop(&mut self) {
        self.shared.clock.lock().unwrap().stopped = true;
        let thread = self.thread.take().unwrap();
        thread.thread().unpark();
        let _ = thread.join();
    }
}
