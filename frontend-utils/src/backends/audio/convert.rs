use cpal::{FromSample, Sample};

#[derive(Clone, Copy)]
pub(super) struct MixerFormat {
    pub channels: usize,
    pub rate: u32,
}

/// Adapt an existing mixer to a replacement device on the PCM producer.
pub(super) struct OutputConverter {
    source: MixerFormat,
    device: MixerFormat,
    phase: u64,
    input: Vec<f32>,
}

impl OutputConverter {
    pub fn new(source: MixerFormat, device: MixerFormat) -> Self {
        Self {
            source,
            device,
            phase: 0,
            input: Vec::new(),
        }
    }

    pub fn fill<T: Sample + FromSample<f32>>(
        &mut self,
        output: &mut [T],
        mix: impl FnOnce(&mut [f32]),
    ) {
        let frames = output.len() / self.device.channels;
        if frames == 0 {
            return;
        }
        let step = u64::from(self.source.rate);
        let rate = u64::from(self.device.rate);
        let end = self.phase + frames as u64 * step;
        let last = end - step;
        // Keep the interpolation neighbour across producer blocks. Consume any
        // skipped source frames as well when downsampling.
        let needed = (last.div_ceil(rate) + 1).max(end / rate) as usize;
        let previous = self.input.len();
        self.input.resize(needed * self.source.channels, 0.);
        if self.input.len() > previous {
            mix(&mut self.input[previous..]);
        }
        for frame in output.chunks_exact_mut(self.device.channels) {
            let index = (self.phase / rate) as usize * self.source.channels;
            let fraction = (self.phase % rate) as f32 / rate as f32;
            let sample = |channel: usize| {
                let index = index + channel.min(self.source.channels - 1);
                let start = self.input[index];
                if fraction == 0. {
                    start
                } else {
                    start + (self.input[index + self.source.channels] - start) * fraction
                }
            };
            frame.fill(T::EQUILIBRIUM);
            if frame.len() == 1 {
                frame[0] = T::from_sample((sample(0) + sample(1)) * 0.5);
            } else {
                frame[0] = T::from_sample(sample(0));
                frame[1] = T::from_sample(sample(1));
            }
            self.phase += step;
        }
        let consumed = (end / rate) as usize * self.source.channels;
        self.input.drain(..consumed);
        self.phase %= rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_and_channel_changes_keep_the_source_clock_across_uneven_blocks() {
        for (from, to) in [(44_100, 48_000), (48_000, 44_100), (48_000, 48_000)] {
            let mut converter = OutputConverter::new(
                MixerFormat {
                    rate: from,
                    channels: 2,
                },
                MixerFormat {
                    rate: to,
                    channels: 1,
                },
            );
            let mut source = 0u32;
            let mut total = 0;
            for count in [1, 37, 255, 3, 240] {
                let mut output = vec![0.; count];
                converter.fill(&mut output, |input| {
                    for frame in input.chunks_exact_mut(2) {
                        frame[0] = source as f32 / 2048.;
                        frame[1] = 0.;
                        source += 1;
                    }
                });
                for sample in output {
                    let expected = (total as f32 * from as f32 / to as f32) / 4096.;
                    assert!((sample - expected).abs() < 1e-6);
                    total += 1;
                }
                assert!(source.abs_diff(total * from / to) <= 2);
            }
        }
    }
}
