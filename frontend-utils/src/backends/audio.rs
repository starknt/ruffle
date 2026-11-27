use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ruffle_core::backend::audio::{
    AudioBackend, AudioMixer, AudioMixerProxy, DecodeError, RegisterError, SoundHandle,
    SoundInstanceHandle, SoundStreamInfo, SoundTransform, swf,
};
use ruffle_core::impl_audio_mixer_backend;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use std::time::{Duration, Instant};

mod convert;
use convert::{MixerFormat, OutputConverter};
mod pcm;
use pcm::PcmWorker;

#[derive(Debug, thiserror::Error)]
pub enum CpalError {
    #[error("No audio devices available")]
    NoDevices,

    #[error("Failed to get default output config")]
    DefaultStream(#[from] cpal::DefaultStreamConfigError),

    #[error("Unsupported sample format {0:?}")]
    UnsupportedSampleFormat(SampleFormat),

    #[error("Couldn't play the audio stream")]
    Play(#[from] cpal::PlayStreamError),

    #[error("Failed to construct audio stream")]
    Build(#[from] cpal::BuildStreamError),

    #[error("Failed to start the PCM producer")]
    Worker(#[from] std::io::Error),
}

pub struct CpalAudioBackend {
    mixer: AudioMixer,
    format: MixerFormat,
    output: Option<DeviceOutput>,
    preferred_device: Option<String>,
    next_retry: Instant,
    playing: bool,
    simulated: bool,
    sample_fraction: f64,
}

struct DeviceOutput {
    _device: cpal::Device,
    stream: cpal::Stream,
    worker: PcmWorker,
    errors: Receiver<cpal::StreamError>,
}

impl CpalAudioBackend {
    pub fn new(preferred_device_name: Option<&str>) -> Result<Self, CpalError> {
        let (device, config) = output_device(preferred_device_name)?;
        let format = MixerFormat {
            channels: usize::from(config.channels().min(2)),
            rate: config.sample_rate().0,
        };
        let mixer = AudioMixer::new(format.channels as u8, format.rate);
        let output = DeviceOutput::open(device, config, &mixer, format, true, false)?;
        Ok(Self {
            mixer,
            format,
            output: Some(output),
            preferred_device: preferred_device_name.map(str::to_owned),
            next_retry: Instant::now(),
            playing: true,
            simulated: false,
            sample_fraction: 0.,
        })
    }

    fn lose_output(&mut self, error: impl std::fmt::Display) {
        tracing::warn!("Audio output unavailable: {error}");
        self.output = None;
        self.next_retry = Instant::now();
    }
}

fn output_device(
    preferred: Option<&str>,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig), CpalError> {
    let device =
        get_suitable_output_device(preferred, &cpal::default_host()).ok_or(CpalError::NoDevices)?;
    let config = device.default_output_config()?;
    Ok((device, config))
}

impl DeviceOutput {
    fn open(
        device: cpal::Device,
        supported: cpal::SupportedStreamConfig,
        mixer: &AudioMixer,
        format: MixerFormat,
        playing: bool,
        simulated: bool,
    ) -> Result<Self, CpalError> {
        let sample_format = supported.sample_format();
        let config = supported.config();
        let (sink, errors) = sync_channel(1);
        let (stream, mut worker) = match sample_format {
            cpal::SampleFormat::F32 => open_stream::<f32>(
                &device,
                &config,
                sink,
                mixer.proxy(),
                format,
                AudioMixerProxy::mix,
            ),
            cpal::SampleFormat::I16 => open_stream::<i16>(
                &device,
                &config,
                sink,
                mixer.proxy(),
                format,
                AudioMixerProxy::mix,
            ),
            cpal::SampleFormat::U16 => open_stream::<u16>(
                &device,
                &config,
                sink,
                mixer.proxy(),
                format,
                |proxy, buffer| {
                    proxy.mix::<i16>(bytemuck::cast_slice_mut(buffer));
                    for sample in buffer {
                        *sample = sample.wrapping_add(32768);
                    }
                },
            ),
            other => return Err(CpalError::UnsupportedSampleFormat(other)),
        }?;
        worker.set_playing(playing);
        worker.set_simulated(simulated);
        if playing {
            stream.play()?;
        }
        Ok(Self {
            _device: device,
            stream,
            worker,
            errors,
        })
    }
}

fn open_stream<T: cpal::SizedSample + cpal::FromSample<f32> + Copy + Send + 'static>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    errors: SyncSender<cpal::StreamError>,
    mixer: AudioMixerProxy,
    format: MixerFormat,
    direct: fn(&AudioMixerProxy, &mut [T]),
) -> Result<(cpal::Stream, PcmWorker), CpalError> {
    let target = MixerFormat {
        channels: usize::from(config.channels),
        rate: config.sample_rate.0,
    };
    let mut conversion = (target.channels != format.channels || target.rate != format.rate)
        .then(|| OutputConverter::new(format, target));
    let (worker, mut output) = PcmWorker::start(
        target.channels,
        target.rate,
        T::EQUILIBRIUM,
        move |buffer| {
            if let Some(converter) = &mut conversion {
                converter.fill(buffer, |input| mixer.mix(input));
            } else {
                direct(&mixer, buffer);
            }
        },
    )?;
    let stream = device.build_output_stream(
        config,
        move |buffer, _| output.fill(buffer),
        move |error| {
            let _ = errors.try_send(error);
        },
        None,
    )?;
    Ok((stream, worker))
}

impl AudioBackend for CpalAudioBackend {
    impl_audio_mixer_backend!(mixer);

    fn set_simulated_output(&mut self, enabled: bool) {
        self.simulated = enabled;
        if let Some(out) = &mut self.output {
            out.worker.set_simulated(enabled);
        }
    }

    fn tick(&mut self) {
        if let Some(error) = self
            .output
            .as_ref()
            .and_then(|out| out.errors.try_recv().ok())
        {
            self.lose_output(error);
        }
        if self.output.is_none() && Instant::now() >= self.next_retry {
            let replacement =
                output_device(self.preferred_device.as_deref()).and_then(|(device, config)| {
                    DeviceOutput::open(
                        device,
                        config,
                        &self.mixer,
                        self.format,
                        self.playing,
                        self.simulated,
                    )
                });
            match replacement {
                Ok(out) => self.output = Some(out),
                Err(error) => {
                    tracing::warn!("Could not reopen audio output: {error}");
                    self.next_retry = Instant::now() + Duration::from_secs(1);
                }
            }
        }
    }
    fn advance_simulated_output(&mut self, millis: f64) {
        let frames = self.sample_fraction + millis * f64::from(self.format.rate) / 1000.0;
        self.sample_fraction = frames.fract();
        let channels = self.format.channels;
        if channels == 0 || channels > 4096 {
            return;
        }
        let mut left = frames as usize;
        let mut discard = [0_i16; 4096];
        while left > 0 {
            let frames = left.min(discard.len() / channels);
            self.mixer.mix(&mut discard[..frames * channels]);
            left -= frames;
        }
    }

    fn play(&mut self) {
        self.playing = true;
        if let Some(out) = &mut self.output {
            out.worker.set_playing(true);
            if let Err(error) = out.stream.play() {
                self.lose_output(error);
            }
        }
    }

    fn pause(&mut self) {
        self.playing = false;
        if let Some(out) = &mut self.output {
            out.worker.set_playing(false);
            if let Err(error) = out.stream.pause() {
                self.lose_output(error);
            }
        }
    }
}

fn get_suitable_output_device(
    preferred_device_name: Option<&str>,
    host: &cpal::Host,
) -> Option<cpal::Device> {
    // First let's check for any user preference...
    if let Some(preferred_device_name) = preferred_device_name
        && let Ok(mut devices) = host.output_devices()
        && let Some(device) =
            devices.find(|device| device.name().ok().as_deref() == Some(preferred_device_name))
    {
        return Some(device);
    }

    // Then let's fall back to the device default
    host.default_output_device()
}

#[cfg(test)]
mod tests {
    use super::pcm::PcmWorker;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn slow_decoder_never_blocks_the_device_and_stale_pcm_cannot_cross_clock_modes() {
        let (entered, mixing) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let mut next = 0i16;
        let (mut worker, mut output) = PcmWorker::start(2, 48000, 0i16, move |samples| {
            if next == 0 {
                entered.send(()).unwrap();
                gate.recv_timeout(Duration::from_secs(2)).unwrap();
            }
            for sample in samples {
                next += 1;
                *sample = next;
            }
        })
        .unwrap();
        let mut frame = [-1; 4];
        output.fill(&mut frame);
        assert_eq!(frame, [0; 4]);
        mixing.recv_timeout(Duration::from_secs(2)).unwrap();
        // The decoder holds the mixer. Both callbacks must still finish before releasing it.
        output.fill(&mut frame);
        assert_eq!(frame, [0; 4]);
        release.send(()).unwrap();
        let wait_ready = |output: &super::pcm::PcmOutput<i16>, samples| {
            let start = std::time::Instant::now();
            while output.available() < samples {
                assert!(start.elapsed() < Duration::from_secs(2));
                std::thread::yield_now();
            }
        };
        wait_ready(&output, 4);
        worker.set_playing(false);
        output.fill(&mut frame);
        assert_eq!(frame, [0; 4]);
        worker.set_playing(true);
        output.fill(&mut frame);
        assert_eq!(frame, [1, 2, 3, 4]);
        wait_ready(&output, 4);
        worker.set_simulated(true);
        output.fill(&mut frame);
        assert_eq!(frame, [0; 4]);
        worker.set_simulated(false);
        output.fill(&mut frame);
        assert_eq!(frame, [0; 4], "no samples from the old clock");
        wait_ready(&output, 4);
        output.fill(&mut frame);
        assert!(frame[0] >= 9);
        assert_eq!(frame[3], frame[0] + 3);
    }
    #[test]
    #[ignore = "requires a native CPAL output, route to private sinks for device-switch testing"]
    fn native_pcm_position_pauses_resumes_and_finishes_on_the_device_clock() {
        use super::{AudioBackend, CpalAudioBackend, swf};
        use cpal::traits::DeviceTrait;
        let mut backend = CpalAudioBackend::new(None).unwrap();
        let rate = 44_100usize;
        let samples: Vec<u8> = (0..rate * 2)
            .flat_map(|i| {
                let phase = (i as f32) * 440.0 * std::f32::consts::TAU / rate as f32;
                ((phase.sin() * 2048.0) as i16).to_le_bytes()
            })
            .collect();
        let sound = backend
            .register_sound(&swf::Sound {
                id: 1,
                format: swf::SoundFormat {
                    compression: swf::AudioCompression::Uncompressed,
                    sample_rate: rate as u16,
                    is_stereo: false,
                    is_16_bit: true,
                },
                num_samples: (rate * 2) as u32,
                data: &samples,
            })
            .unwrap();
        let instance = backend
            .start_sound(
                sound,
                &swf::SoundInfo {
                    event: swf::SoundEvent::Start,
                    in_sample: None,
                    out_sample: None,
                    num_loops: 1,
                    envelope: None,
                },
            )
            .unwrap();
        let start = std::time::Instant::now();
        let mut previous = 0.;
        let mut paused = false;
        let mut simulated = false;
        while let Some(position) = backend.get_sound_position(instance) {
            assert!(position >= previous);
            previous = position;
            assert!(
                start.elapsed() < Duration::from_secs(8),
                "native audio clock stopped at {position}"
            );
            if !paused && position >= 900. {
                backend.pause();
                let stopped = backend.get_sound_position(instance).unwrap();
                let (failed_device, errors) = std::sync::mpsc::sync_channel(1);
                backend.output.as_mut().unwrap().errors = errors;
                failed_device
                    .send(cpal::StreamError::DeviceNotAvailable)
                    .unwrap();
                backend.tick();
                assert!(backend.output.is_some());
                assert!(
                    matches!(
                        failed_device.try_send(cpal::StreamError::DeviceNotAvailable),
                        Err(std::sync::mpsc::TrySendError::Disconnected(_))
                    ),
                    "a failed device stream must be replaced on the player thread"
                );
                std::thread::sleep(Duration::from_millis(150));
                assert_eq!(backend.get_sound_position(instance), Some(stopped));
                backend.play();
                paused = true;
            }
            if !simulated && position >= 1250. {
                backend.set_simulated_output(true);
                let before = backend.get_sound_position(instance).unwrap();
                let (device, _) = super::output_device(None).unwrap();
                let alternate = if backend.format.rate == 44_100 {
                    48_000
                } else {
                    44_100
                };
                let config = device
                    .supported_output_configs()
                    .unwrap()
                    .find_map(|range| {
                        (matches!(
                            range.sample_format(),
                            cpal::SampleFormat::F32 | cpal::SampleFormat::I16
                        ) && range.channels() == 2
                            && range.min_sample_rate().0 <= alternate
                            && alternate <= range.max_sample_rate().0)
                            .then(|| range.with_sample_rate(cpal::SampleRate(alternate)))
                    })
                    .expect("the test output must offer both 44.1 and 48 kHz");
                backend.output = None;
                backend.output = Some(
                    super::DeviceOutput::open(
                        device,
                        config,
                        &backend.mixer,
                        backend.format,
                        true,
                        true,
                    )
                    .unwrap(),
                );
                std::thread::sleep(Duration::from_millis(50));
                assert_eq!(backend.get_sound_position(instance), Some(before));
                backend.advance_simulated_output(75.);
                let advance = backend.get_sound_position(instance).unwrap() - before;
                assert!(
                    (73. ..=77.).contains(&advance),
                    "simulated advance {advance} ms"
                );
                backend.set_simulated_output(false);
                simulated = true;
            }
            std::thread::sleep(Duration::from_millis(10));
            backend.tick();
        }
        assert!(paused && simulated && previous >= 1500.);
        assert!(start.elapsed() >= Duration::from_millis(1700));
    }
}
