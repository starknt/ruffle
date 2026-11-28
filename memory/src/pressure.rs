use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PressureLevel {
    #[default]
    Normal,
    Moderate,
    High,
    Critical,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PressureReading {
    pub level: PressureLevel,
    /// Monotonically identifies a new critical episode for each consumer.
    pub critical_epoch: u64,
}

#[derive(Debug, Default)]
pub struct PressureState(PressureReading);

impl PressureState {
    pub fn reading(&self) -> PressureReading {
        self.0
    }

    pub fn update(&mut self, used: u64, budget: u64, floor: PressureLevel) -> PressureLevel {
        let previous = self.0.level;
        let mut level = floor;
        for (candidate, threshold) in [
            (PressureLevel::Moderate, 65u64),
            (PressureLevel::High, 80),
            (PressureLevel::Critical, 95),
        ] {
            let threshold = if previous >= candidate {
                threshold - 5
            } else {
                threshold
            };
            if u128::from(used) * 100 >= u128::from(budget.max(1)) * u128::from(threshold) {
                level = level.max(candidate);
            }
        }
        if level == PressureLevel::Critical && previous != level {
            self.0.critical_epoch += 1;
        }
        self.0.level = level;
        level
    }
}
