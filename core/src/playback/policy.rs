//! Host eligibility and wall-clock class policy, independent of simulation time.
use crate::string::WStr;
use std::{cell::Cell, rc::Rc};

pub trait PlaybackPolicy {
    fn observe_definition(&self, namespace: &WStr, name: &WStr, movie: &[u8]);
    fn allows_acceleration(&self) -> bool;
}

#[derive(Default)]
pub(crate) struct HostPlayback {
    pub(crate) policy: Option<Rc<dyn PlaybackPolicy>>,
    pub(crate) wall_classes: &'static [(&'static str, &'static str)],
    modules: Option<ModuleCheck>,
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

impl HostPlayback {
    pub(crate) fn install_module_check(
        &mut self,
        rules: &'static [ModuleRule],
        verify: fn(&[u8], &[&str]) -> bool,
    ) {
        self.policy = None;
        self.modules = Some(ModuleCheck {
            rules,
            verify,
            seen: std::cell::RefCell::new(vec![false; rules.len()]),
            mismatch: Cell::default(),
        });
    }
    pub(crate) fn check_definition(&self, name: crate::avm2::QName<'_>, data: &[u8]) {
        if let Some(policy) = &self.policy {
            if let Some(uri) = name.namespace().as_uri_opt() {
                policy.observe_definition(uri.as_wstr(), name.local_name().as_wstr(), data);
            }
            return;
        }
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
    pub(crate) fn install(&mut self, policy: Rc<dyn PlaybackPolicy>) {
        self.modules = None;
        self.policy = Some(policy);
    }

    pub(crate) fn allows_acceleration(&self) -> bool {
        self.policy.as_ref().map_or_else(
            || self.unverified_module().is_none(),
            |p| p.allows_acceleration(),
        )
    }
}
