//! CPU creation timings and actual native draw/dispatch use, scoped to one device.
use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    collections::BTreeMap,
    fmt::{self, Write},
    sync::Arc,
    time::Instant,
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Family {
    TwoD,
    Copy,
    Filter,
    Stage3d,
    PixelBender,
    Compositor,
    Upscale,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    Shader,
    Render,
    Compute,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Signature([u8; 32]);

/// Input consists of content and descriptor state under this renderer ABI.
pub fn signature(input: fmt::Arguments<'_>) -> Signature {
    struct Sink(Sha256);
    impl Write for Sink {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            self.0.update(s.as_bytes());
            Ok(())
        }
    }
    let mut writer = Sink(Sha256::new());
    writer
        .write_fmt(input)
        .expect("digest writes are infallible");
    Signature(writer.0.finalize().into())
}

impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Frame {
    pub device: u64,
    pub session: u64,
    pub frame_id: u64,
}

thread_local! { static FRAME: Cell<Option<Frame>> = const { Cell::new(None) }; }
pub struct FrameScope(Option<Frame>, std::marker::PhantomData<std::rc::Rc<()>>);
pub fn frame_scope(frame: Frame) -> FrameScope {
    FrameScope(FRAME.replace(Some(frame)), Default::default())
}
impl Drop for FrameScope {
    fn drop(&mut self) {
        FRAME.set(self.0);
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreationSample {
    frame: Option<Frame>,
    calls: u64,
    ns: u64,
}

#[derive(Default, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Counts {
    creates: u64,
    creation_ns: u64,
    uses: u64,
    cache_hits: u64,
    resident: u64,
    peak_resident: u64,
    first_use: Option<Frame>,
    creation_samples: Vec<CreationSample>,
    unlisted_calls: u64,
}

pub struct Variant {
    device: u64,
    counts: Mutex<Counts>,
}

impl Variant {
    fn frame(&self) -> Option<Frame> {
        FRAME.get().filter(|f| f.device == self.device)
    }

    pub fn measure<T>(&self, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let value = f();
        let ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let frame = self.frame();
        let mut c = self.counts.lock();
        c.creates += 1;
        c.creation_ns = c.creation_ns.saturating_add(ns);
        if let Some(last) = c
            .creation_samples
            .last_mut()
            .filter(|last| last.frame == frame)
        {
            last.calls += 1;
            last.ns = last.ns.saturating_add(ns);
        } else if c.creation_samples.len() < 64 {
            c.creation_samples.push(CreationSample {
                frame,
                calls: 1,
                ns,
            });
        } else {
            c.unlisted_calls += 1;
        }
        value
    }

    pub fn instance(self: &Arc<Self>) -> Instance {
        let mut c = self.counts.lock();
        c.resident += 1;
        c.peak_resident = c.peak_resident.max(c.resident);
        Instance(self.clone())
    }

    pub fn used(&self) {
        let mut c = self.counts.lock();
        if c.uses == 0 {
            c.first_use = self.frame();
        }
        c.uses += 1;
    }

    pub fn hit(&self) {
        self.counts.lock().cache_hits += 1;
    }
}

pub struct Instance(Arc<Variant>);
impl Instance {
    pub fn variant(&self) -> &Arc<Variant> {
        &self.0
    }
}
impl Drop for Instance {
    fn drop(&mut self) {
        self.0.counts.lock().resident -= 1;
    }
}

pub(crate) struct ObjectInfo {
    pub key: Signature,
    pub instance: Option<Instance>,
}

#[derive(Default)]
pub struct Profile {
    entries: Mutex<BTreeMap<(Family, Kind, Signature), Arc<Variant>>>,
    overflow: std::sync::atomic::AtomicU64,
}

impl Profile {
    pub fn variant(
        &self,
        device: u64,
        family: Family,
        kind: Kind,
        key: Signature,
    ) -> Option<Arc<Variant>> {
        let mut entries = self.entries.lock();
        if entries.len() >= 4096 && !entries.contains_key(&(family, kind, key)) {
            self.overflow
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
        Some(
            entries
                .entry((family, kind, key))
                .or_insert_with(|| {
                    Arc::new(Variant {
                        device,
                        counts: Default::default(),
                    })
                })
                .clone(),
        )
    }

    pub fn snapshot(&self, device: u64) -> serde_json::Value {
        let variants: Vec<_> = self
            .entries
            .lock()
            .iter()
            .map(|((family, kind, key), v)| {
                #[derive(Serialize)]
                struct Row {
                    family: Family,
                    kind: Kind,
                    key: String,
                    #[serde(flatten)]
                    counts: Counts,
                }
                Row {
                    family: *family,
                    kind: *kind,
                    key: key.to_string(),
                    counts: v.counts.lock().clone(),
                }
            })
            .collect();
        serde_json::json!({"device":device, "variants":variants, "overflow": self.overflow.load(std::sync::atomic::Ordering::Relaxed)})
    }
}
