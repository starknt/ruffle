use std::ops::Range;
use std::panic::catch_unwind;
use std::path::Path;
use std::sync::Arc;

use image::RgbaImage;
use ruffle_core::PlayerBuilder;
use ruffle_core::PlayerSession;
use ruffle_core::backend::locale::DeterministicLocaleBackend;
use ruffle_core::limits::ExecutionLimit;
use ruffle_core::tag_utils::movie_from_path;
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
            .with_viewport_dimensions(width, height, self.size.scale)
            .build_session();

        Ok(MovieExport {
            frames: self.frames.capture_range(session.player(), self.skipframes),
            session,
            force_play: self.force_play,
        })
    }
}

pub struct MovieExport {
    session: PlayerSession,
    pub frames: Range<u64>,
    force_play: bool,
}

impl MovieExport {
    pub fn run_frame(&mut self) {
        self.session.poll_tasks();
        let player = self.session.player();
        if self.force_play {
            player.force_root_clip_play();
        }

        player.lock().unwrap().preload(&mut ExecutionLimit::none());

        player.lock().unwrap().run_frame();
        self.session.poll_tasks();
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
