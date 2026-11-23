//! Host eligibility and wall-clock class policy, independent of simulation time.
use crate::string::WStr;
use std::rc::Rc;

pub trait PlaybackPolicy {
    fn observe_definition(&self, _namespace: &WStr, _name: &WStr, _movie: &[u8]) {}
    fn allows_acceleration(&self) -> bool;
}

#[derive(Default)]
pub(crate) struct HostPlayback {
    policy: Option<Rc<dyn PlaybackPolicy>>,
    pub(crate) wall_classes: &'static [(&'static str, &'static str)],
}

impl HostPlayback {
    pub(crate) fn check_definition(&self, name: crate::avm2::QName<'_>, data: &[u8]) {
        if let Some(policy) = &self.policy
            && let Some(uri) = name.namespace().as_uri_opt()
        {
            policy.observe_definition(uri.as_wstr(), name.local_name().as_wstr(), data);
        }
    }

    pub(crate) fn install(&mut self, policy: Rc<dyn PlaybackPolicy>) {
        self.policy = Some(policy);
    }

    pub(crate) fn allows_acceleration(&self) -> bool {
        self.policy.as_ref().is_none_or(|p| p.allows_acceleration())
    }
}
