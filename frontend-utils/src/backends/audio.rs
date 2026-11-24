use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ruffle_core::backend::audio::{
    AudioBackend, AudioMixer, DecodeError, RegisterError, SoundHandle, SoundInstanceHandle,
    SoundStreamInfo, SoundTransform, swf,
};
use ruffle_core::impl_audio_mixer_backend;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

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
    #[expect(dead_code)]
    device: cpal::Device,
    config: cpal::StreamConfig,
    stream: cpal::Stream,
    mixer: AudioMixer,
    worker: PcmWorker,
    errors: Receiver<cpal::StreamError>,
    sample_fraction: f64,
}

impl CpalAudioBackend {
    pub fn new(preferred_device_name: Option<&str>) -> Result<Self, CpalError> {
        // Create CPAL audio device.
        let host = cpal::default_host();
        let device =
            get_suitable_output_device(preferred_device_name, &host).ok_or(CpalError::NoDevices)?;

        // Create audio stream for device.
        let config = device
            .default_output_config()
            .map_err(CpalError::DefaultStream)?;
        let sample_format = config.sample_format();
        let config = cpal::StreamConfig::from(config);
        let mixer = AudioMixer::new(config.channels as u8, config.sample_rate.0);

        let proxy = mixer.proxy();
        let (error_sink, errors) = sync_channel(1);
        let (stream, worker) = match sample_format {
            cpal::SampleFormat::F32 => {
                open_stream(&device, &config, error_sink, 0.0f32, move |b| proxy.mix(b))
            }
            cpal::SampleFormat::I16 => {
                open_stream(&device, &config, error_sink, 0i16, move |b| proxy.mix(b))
            }
            cpal::SampleFormat::U16 => {
                open_stream(&device, &config, error_sink, 32768u16, move |buffer| {
                    proxy.mix::<i16>(bytemuck::cast_slice_mut(buffer));
                    for sample in buffer {
                        *sample = sample.wrapping_add(32768);
                    }
                })
            }
            _ => return Err(CpalError::UnsupportedSampleFormat(sample_format)),
        }?;

        stream.play().map_err(CpalError::Play)?;

        Ok(Self {
            device,
            config,
            worker,
            errors,
            sample_fraction: 0.0,
            stream,
            mixer,
        })
    }
}

fn open_stream<T: cpal::SizedSample + Copy + Send + 'static>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    errors: SyncSender<cpal::StreamError>,
    silence: T,
    mix: impl FnMut(&mut [T]) + Send + 'static,
) -> Result<(cpal::Stream, PcmWorker), CpalError> {
    let (worker, mut output) = PcmWorker::start(
        usize::from(config.channels),
        config.sample_rate.0,
        silence,
        mix,
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
        self.worker.set_simulated(enabled);
    }

    fn tick(&mut self) {
        if let Ok(error) = self.errors.try_recv() {
            tracing::error!("Audio stream error: {error}");
        }
    }
    fn advance_simulated_output(&mut self, millis: f64) {
        let frames = self.sample_fraction + millis * f64::from(self.config.sample_rate.0) / 1000.0;
        self.sample_fraction = frames.fract();
        let channels = usize::from(self.config.channels);
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
        self.worker.set_playing(true);
        self.stream.play().expect("Error trying to resume CPAL audio stream. This feature may not be supported by your audio device.");
    }

    fn pause(&mut self) {
        self.stream.pause().expect("Error trying to pause CPAL audio stream. This feature may not be supported by your audio device.");
        self.worker.set_playing(false);
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
        while let Some(position) = backend.get_sound_position(instance) {
            assert!(position >= previous);
            previous = position;
            assert!(
                start.elapsed() < Duration::from_secs(8),
                "native audio clock stopped at {position}"
            );
            if !paused && position >= 650. {
                backend.pause();
                let stopped = backend.get_sound_position(instance).unwrap();
                std::thread::sleep(Duration::from_millis(150));
                assert_eq!(backend.get_sound_position(instance), Some(stopped));
                backend.play();
                paused = true;
            }
            std::thread::sleep(Duration::from_millis(10));
            backend.tick();
        }
        assert!(paused && previous >= 1500.);
        assert!(start.elapsed() >= Duration::from_millis(1700));
    }
}
