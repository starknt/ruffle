use std::ops::Range;
use std::panic::catch_unwind;
use std::path::Path;
use std::sync::Arc;

use image::RgbaImage;
use ruffle_core::PlayerSession;
use ruffle_core::backend::locale::DeterministicLocaleBackend;
use ruffle_core::limits::ExecutionLimit;
use ruffle_core::tag_utils::movie_from_path;
use ruffle_core::{FloatDuration, PlayerBuilder};
use ruffle_render_wgpu::backend::{
    WgpuRenderBackend, create_wgpu_instance, request_adapter_and_device,
};
use ruffle_render_wgpu::descriptors::Descriptors;

use anyhow::Result;
use anyhow::anyhow;
use ruffle_render_wgpu::target::TextureTarget;
use ruffle_render_wgpu::wgpu;

use crate::cli::FrameSelection;
use crate::cli::Opt;
use crate::cli::SizeOpt;
use crate::player_ext::PlayerExporterExt;

pub struct Exporter {
    descriptors: Arc<Descriptors>,
    size: SizeOpt,
    skipframes: u32,
    frames: FrameSelection,
    force_play: bool,
}

impl Exporter {
    pub fn new(opt: &Opt) -> Result<Self> {
        let instance =
            create_wgpu_instance(opt.graphics.into(), wgpu::BackendOptions::default(), None);
        let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
            opt.graphics.into(),
            &instance,
            None,
            opt.power.into(),
        ))
        .map_err(|e| anyhow!(e.to_string()))?;

        let descriptors = Arc::new(Descriptors::new(instance, adapter, device, queue)?);

        Ok(Self {
            descriptors,
            size: opt.size,
            skipframes: opt.skipframes,
            frames: opt.frames,
            force_play: opt.force_play,
        })
    }

    pub fn start_exporting_movie(&self, swf_path: &Path) -> Result<MovieExport> {
        let movie = movie_from_path(swf_path, None).map_err(|e| anyhow!(e.to_string()))?;

        let width = self
            .size
            .width
            .map(f64::from)
            .unwrap_or_else(|| movie.width().to_pixels());
        let width = (width * self.size.scale).round() as u32;

        let height = self
            .size
            .height
            .map(f64::from)
            .unwrap_or_else(|| movie.height().to_pixels());
        let height = (height * self.size.scale).round() as u32;

        let target = TextureTarget::new(&self.descriptors.device, (width, height))
            .map_err(|e| anyhow!(e.to_string()))?;
        let session = PlayerBuilder::new()
            .with_renderer(
                WgpuRenderBackend::new(self.descriptors.clone(), target)
                    .map_err(|e| anyhow!(e.to_string()))?,
            )
            .with_locale(DeterministicLocaleBackend::default())
            .with_movie(movie)
            .with_autoplay(true)
            .with_viewport_dimensions(width, height, self.size.scale)
            .build_session();

        Ok(MovieExport {
            sample_time: None,
            frames: self.frames.capture_range(session.player(), self.skipframes),
            session,
            force_play: self.force_play,
        })
    }
}

pub struct MovieExport {
    session: PlayerSession,
    sample_time: Option<FloatDuration>,
    pub frames: Range<u64>,
    force_play: bool,
}

impl MovieExport {
    pub fn run_frame(&mut self) -> Result<()> {
        self.session.poll_tasks();
        if self.force_play {
            self.session.player().force_root_clip_play();
        }
        let mut player = self.session.lock().unwrap();
        player.preload(&mut ExecutionLimit::none());
        let rate = player.frame_rate();
        if let Some(previous) = self.sample_time
            && rate.is_finite()
            && rate > 0.0
        {
            let end = previous + FloatDuration::from_millis(1000.0 / rate);
            let mut time = previous.to_std();
            while time < end.to_std() {
                let step = player.next_sample_step(end.to_std() - time);
                time += step;
                player.set_sample_time(time);
                player.tick(FloatDuration::from_std(step))?;
                drop(player);
                self.session.poll_tasks();
                player = self.session.lock().unwrap();
            }
            self.sample_time = Some(end);
        } else {
            let time = self.sample_time.get_or_insert(FloatDuration::ZERO);
            player.set_sample_time(time.to_std());
            player.run_frame();
        }
        drop(player);
        self.session.poll_tasks();
        Ok(())
    }

    pub fn capture_frame(&self) -> Result<RgbaImage> {
        let player = self.session.player();
        let image = || {
            if player.lock().unwrap().render()? != ruffle_core::FrameOutcome::Submitted {
                return Ok(None);
            }
            player.capture_frame()
        };
        match catch_unwind(image) {
            Ok(Ok(Some(image))) => Ok(image),
            Ok(Ok(None)) => Err(anyhow!("No frame captured")),
            Ok(Err(error)) => Err(error.into()),
            Err(e) => Err(anyhow!("{e:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn headless_sibling_reads_keep_the_opened_file_inside_the_input_directory() {
        use ruffle_core::backend::navigator::{FetchReason, Request};
        use std::{fs, os::unix::fs::symlink};
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("movie");
        fs::create_dir(&directory).unwrap();
        let file = directory.join("test.swf");
        fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/tests/swfs/avm1/asnew/test.swf"),
            &file,
        )
        .unwrap();
        fs::write(directory.join("sibling"), b"inside").unwrap();
        fs::write(temp.path().join("other"), b"outside").unwrap();
        let alias = directory.join("alias");
        symlink(directory.join("sibling"), &alias).unwrap();
        let session = PlayerBuilder::new()
            .with_movie(movie_from_path(file, None).unwrap())
            .build();
        let read = || {
            session
                .lock()
                .unwrap()
                .fetch(Request::get("alias".into()), FetchReason::UrlLoader)
        };
        futures::executor::block_on(async {
            let mut response = read().await.map_err(|e| e.error).unwrap();
            fs::remove_file(&alias).unwrap();
            symlink(temp.path().join("other"), &alias).unwrap();
            assert!(read().await.is_err());
            assert_eq!(response.next_chunk().await.unwrap().unwrap(), b"inside");
            assert_eq!(response.body().await.unwrap(), b"inside");
        });
    }

    #[test]
    fn skipped_and_captured_frames_advance_the_same_authored_clock() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/tests/swfs/avm1/asnew/test.swf");
        let movie = movie_from_path(path, None).unwrap();
        let period = 1000.0 / movie.frame_rate().to_f64();
        let session = PlayerBuilder::new()
            .with_movie(movie)
            .with_autoplay(true)
            .build();
        let mut export = MovieExport {
            sample_time: None,
            session,
            frames: 2..4,
            force_play: false,
        };
        for index in 0..4 {
            export.run_frame().unwrap();
            let time = export.session.lock().unwrap().playback_time_ms();
            assert!(
                (time - f64::from(index) * period).abs() < 0.001,
                "sample {index}: {time}"
            );
        }
    }
}
