//! The simulation clock belongs to one player. Network and AVM execution watchdogs use wall time.
use std::cell::Cell;
use web_time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum PlaybackRate {
    #[default]
    Normal = 1,
    Double = 2,
    Five = 5,
    Ten = 10,
}
impl TryFrom<u8> for PlaybackRate {
    type Error = &'static str;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Normal),
            2 => Ok(Self::Double),
            5 => Ok(Self::Five),
            10 => Ok(Self::Ten),
            _ => Err("Playback rate must be 1, 2, 5 or 10"),
        }
    }
}

/// A frontend identifies only the definitions whose bytecode it has audited.
pub struct ModuleRule {
    pub namespace: &'static str,
    pub class: &'static str,
    pub fingerprints: &'static [&'static str],
}
struct ModuleCheck {
    rules: &'static [ModuleRule],
    verify: fn(&[u8], &[&str]) -> bool,
    seen: std::cell::RefCell<Vec<bool>>,
    mismatch: Cell<Option<usize>>,
}

#[derive(Default)]
pub struct PlaybackClock {
    modules: Option<ModuleCheck>,
    pub(crate) rate: PlaybackRate,
    pub(crate) elapsed: Option<f64>,
    pub(crate) credit: f64,
    pub(crate) executing_since: Option<Instant>,
    last_read: Cell<f64>,
    pub(crate) wall_classes: &'static [(&'static str, &'static str)],
    pub(crate) wall_frame_credit: f64,
    pub(crate) batching_input: bool,
    pub(crate) callback_clock: Cell<Option<(usize, bool)>>,
    pub(crate) wall_deadline: Option<f64>,
    pub(crate) action_base: f64,
    pub(crate) wall_frame_due: bool,
}
impl PlaybackClock {
    pub(crate) fn install_module_check(
        &mut self,
        rules: &'static [ModuleRule],
        verify: fn(&[u8], &[&str]) -> bool,
    ) {
        self.modules = Some(ModuleCheck {
            rules,
            verify,
            seen: std::cell::RefCell::new(vec![false; rules.len()]),
            mismatch: Cell::default(),
        });
    }
    pub(crate) fn check_definition(&self, name: crate::avm2::QName<'_>, data: &[u8]) {
        let Some(check) = &self.modules else {
            return;
        };
        if check.mismatch.get().is_some() {
            return;
        }
        let Some(uri) = name.namespace().as_uri_opt() else {
            return;
        };
        if let Some((i, rule)) = check.rules.iter().enumerate().find(|(_, r)| {
            name.local_name().as_wstr() == r.class.as_bytes()
                && uri.as_wstr() == r.namespace.as_bytes()
        }) {
            if (check.verify)(data, rule.fingerprints) {
                check.seen.borrow_mut()[i] = true;
            } else {
                check.mismatch.set(Some(i));
            }
        }
    }
    pub(crate) fn unverified_module(&self) -> Option<(&'static ModuleRule, bool)> {
        let check = self.modules.as_ref()?;
        if let Some(i) = check.mismatch.get() {
            return Some((&check.rules[i], true));
        }
        check
            .seen
            .borrow()
            .iter()
            .position(|seen| !*seen)
            .map(|i| (&check.rules[i], false))
    }
    pub(crate) fn begin_action(&mut self) {
        self.action_base = self.last_read.get().max(self.elapsed.unwrap_or(0.));
        self.executing_since = Some(Instant::now());
    }
    pub(crate) fn read(&self, start: Instant) -> f64 {
        let Some(elapsed) = self.elapsed else {
            return start.elapsed().as_secs_f64() * 1000.0;
        };
        // A busy loop in a single AS call still observes time moving. Never rewind the next read.
        let now = if self.executing_since.is_some() {
            self.action_base.max(elapsed)
        } else {
            elapsed
        } + self
            .executing_since
            .map_or(0.0, |at| at.elapsed().as_secs_f64() * 1000.0);
        let now = now.max(self.last_read.get());
        self.last_read.set(now);
        now
    }
}
