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

#[derive(Default)]
pub struct PlaybackClock {
    pub(crate) rate: PlaybackRate,
    pub(crate) elapsed: Option<f64>,
    pub(crate) credit: f64,
    pub(crate) executing_since: Option<Instant>,
    last_read: Cell<f64>,
    pub(crate) wall_classes: &'static [(&'static str, &'static str)],
    pub(crate) wall_frame_credit: f64,
    pub(crate) mixed_timer: Cell<bool>,
    pub(crate) wall_frame_due: bool,
}
impl PlaybackClock {
    pub(crate) fn read(&self, start: Instant) -> f64 {
        let Some(elapsed) = self.elapsed else {
            return start.elapsed().as_secs_f64() * 1000.0;
        };
        // A busy loop in a single AS call still observes time moving. Never rewind the next read.
        let now = elapsed
            + self
                .executing_since
                .map_or(0.0, |at| at.elapsed().as_secs_f64() * 1000.0);
        let now = now.max(self.last_read.get());
        self.last_read.set(now);
        now
    }
}
