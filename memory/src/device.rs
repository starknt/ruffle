//! Device totals are queried from the owning ledger. A process registry keeps only weak references.
use super::PressureReading;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeviceSnapshot {
    pub id: u64,
    pub budget_bytes: u64,
    pub logical_bytes: u64,
    pub reserved_bytes: u64,
    pub active_bytes: u64,
    pub idle_bytes: u64,
    pub retiring_bytes: u64,
    pub in_flight_bytes: u64,
    pub host_sampling_bytes: u64,
    pub peak_bytes: u64,
    pub pressure: PressureReading,
    pub classes: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceSnapshot {
    pub id: u64,
    pub owner_id: u64,
    #[serde(flatten)]
    pub symbol: Option<SymbolId>,
    pub owner_kind: &'static str,
    pub kind: &'static str,
    pub bytes: u64,
    pub last_submission: u64,
    pub state: &'static str,
}

/// Native driver allocator counters. Some backends do not expose them.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AllocatorSnapshot {
    pub allocated_bytes: u64,
    pub reserved_bytes: u64,
}

pub trait DeviceMemory: Send + Sync + std::fmt::Debug {
    fn id(&self) -> u64;
    fn snapshot(&self) -> DeviceSnapshot;
    fn resources(&self) -> Vec<ResourceSnapshot>;
}

/// Stable origin of an embedded asset. The device ledger stores only these numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SymbolId {
    pub movie_id: u64,
    pub character_id: u16,
}
thread_local! {
    static SYMBOL: std::cell::Cell<Option<SymbolId>> = const { std::cell::Cell::new(None) };
}
impl SymbolId {
    pub fn current() -> Option<Self> {
        SYMBOL.get()
    }
    pub fn enter<R>(symbol: Option<Self>, work: impl FnOnce() -> R) -> R {
        struct Restore(Option<SymbolId>);
        impl Drop for Restore {
            fn drop(&mut self) {
                SYMBOL.set(self.0);
            }
        }
        let _restore = Restore(SYMBOL.replace(symbol));
        work()
    }
}
