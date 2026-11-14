//! Anonymous traversal counts. Compiled only for diagnostic builds; these
//! counts explain missed cache opportunities, not CPU time or retained memory.
use std::{cell::Cell, marker::PhantomData, rc::Rc};

macro_rules! names {
    ($enum:ident, $names:ident, $($variant:ident => $name:literal),+ $(,)?) => {
        #[derive(Clone, Copy)]
        pub enum $enum { $($variant),+ }
        pub const $names: &[&str] = &[$($name),+];
    };
}

names!(Phase, PHASE_NAMES,
    Frame => "frame",
    OrphanConstruct => "orphanConstruct",
    StageConstruct => "stageConstruct",
    StageScripts => "stageScripts",
    OrphanScripts => "orphanScripts",
);

names!(Count, COUNT_NAMES,
    ConstructVisit => "constructVisit",
    ConstructHit => "constructHit",
    ConstructSettled => "constructSettled",
    Uninitialized => "uninitialized",
    NoObject => "noObject",
    QueuedGoto => "queuedGoto",
    RevisionChanged => "revisionChanged",
    ChildrenUnsettled => "childrenUnsettled",
    RegisteredScripts => "scriptSlotsNonempty",
    OutsideConstruct => "outsideConstruct",
    NewObject => "newObject",
    ManualLoad => "uninitializedScriptPlaced",
    NoTimeline => "timelineNotLoaded",
    NotAvm2 => "notAvm2",
    NotOptimized => "notOptimized",
    ChildMovieClip => "childMovieClipUnsettled",
    ChildButton => "childButtonBoundary",
    ChildLoader => "childLoaderBoundary",
    ChildLeaf => "childLeafUnsettled",
    ChildManual => "childManual",
    ChildRecursive => "childRecursive",
    ScriptsVisit => "scriptsVisit",
    ScriptsHit => "scriptsHit",
    ScriptsSettled => "scriptsSettled",
    ScriptsConstructionDirty => "scriptsConstructionDirty",
    ScriptsPending => "scriptsPending",
    ScriptsExecuting => "scriptsExecuting",
    ScriptsGoto => "scriptsGoto",
    ScriptsChildActive => "scriptsChildActive",
    ScriptsRevisionChanged => "scriptsRevisionChanged",
    OrphanEntries => "orphanEntries",
    OrphanMovieClip => "orphanMovieClip",
    OrphanOther => "orphanOther",
    Invalidations => "invalidations",
    InvalidatedAncestors => "invalidatedAncestors",
    PendingChanged => "pendingChanged",
    LastQueuedChanged => "lastQueuedChanged",
    WorklistVisit => "worklistVisit",
    WorklistEntries => "worklistEntries",
    WorklistSkipped => "worklistSkipped",
    WorklistWake => "worklistWake",
    WorklistButtonWake => "worklistButtonWake",
    WorklistFallback => "worklistFallback",
);

#[derive(Clone, Debug)]
pub struct Counters {
    pub values: [[u64; COUNT_NAMES.len()]; PHASE_NAMES.len()],
}

impl Default for Counters {
    fn default() -> Self {
        Self {
            values: [[0; COUNT_NAMES.len()]; PHASE_NAMES.len()],
        }
    }
}

thread_local! {
    static PHASE: Cell<Phase> = const { Cell::new(Phase::Frame) };
}

pub struct Scope(Phase, PhantomData<Rc<()>>);

pub fn scope(phase: Phase) -> Scope {
    Scope(PHASE.replace(phase), PhantomData)
}

impl Drop for Scope {
    fn drop(&mut self) {
        PHASE.set(self.0);
    }
}

pub fn add(count: Count, amount: u64) {
    super::CURRENT.with_borrow(|metrics| {
        if let Some(metrics) = metrics {
            metrics.0.borrow_mut().lifecycle.values[PHASE.get() as usize][count as usize] += amount;
        }
    });
}
