use crate::decoder::VideoDecoder;
use ruffle_render::backend::RenderBackend;
use ruffle_render::bitmap::{BitmapHandle, BitmapInfo, PixelRegion};
use ruffle_video::VideoStreamHandle;
use ruffle_video::backend::VideoBackend;
use ruffle_video::error::Error;
use ruffle_video::frame::{EncodedFrame, FrameDependency};
use swf::{VideoCodec, VideoDeblocking};

/// Software video backend that proxies to CPU-only codec implementations that
/// ship with Ruffle.
pub struct SoftwareVideoBackend;

impl Default for SoftwareVideoBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftwareVideoBackend {
    pub fn new() -> Self {
        Self
    }
}

impl VideoBackend for SoftwareVideoBackend {
    #[allow(unreachable_code, unused_variables)]
    fn register_video_stream(
        &mut self,
        _num_frames: u32,
        size: (u16, u16),
        codec: VideoCodec,
        filter: VideoDeblocking,
    ) -> Result<VideoStreamHandle, Error> {
        let decoder: Box<dyn VideoDecoder> = match codec {
            #[cfg(feature = "h263")]
            VideoCodec::H263 => Box::new(crate::decoder::h263::H263Decoder::new(filter)),
            #[cfg(feature = "vp6")]
            VideoCodec::Vp6 => Box::new(crate::decoder::vp6::Vp6Decoder::new(false, size)),
            #[cfg(feature = "vp6")]
            VideoCodec::Vp6WithAlpha => Box::new(crate::decoder::vp6::Vp6Decoder::new(true, size)),
            #[cfg(feature = "screenvideo")]
            VideoCodec::ScreenVideo => Box::new(crate::decoder::screen::ScreenVideoDecoder::new()),
            other => return Err(Error::UnsupportedCodec(other)),
        };
        let stream = VideoStream::new(decoder);
        Ok(VideoStreamHandle::new(stream))
    }

    fn preload_video_stream_frame(
        &mut self,
        stream: VideoStreamHandle,
        encoded_frame: EncodedFrame<'_>,
    ) -> Result<FrameDependency, Error> {
        let mut stream = stream.borrow_mut::<VideoStream>()?;
        let stream = &mut *stream;

        stream.decoder.preload_frame(encoded_frame)
    }

    fn configure_video_stream_decoder(
        &mut self,
        _stream: VideoStreamHandle,
        _configuration_data: &[u8],
    ) -> Result<(), Error> {
        // None of the software decoders require configuration.
        Ok(())
    }

    fn decode_video_stream_frame(
        &mut self,
        stream: VideoStreamHandle,
        encoded_frame: EncodedFrame<'_>,
        renderer: &mut dyn RenderBackend,
    ) -> Result<BitmapInfo, Error> {
        let mut stream = stream.borrow_mut::<VideoStream>()?;
        let stream = &mut *stream;

        stream.decoder.decode_frame(encoded_frame, |frame| {
            let width = frame.width();
            let height = frame.height();

            let handle = if let Some(bitmap) = stream.bitmap.clone() {
                renderer.update_texture(
                    &bitmap,
                    frame,
                    PixelRegion::for_whole_size(width, height),
                )?;
                bitmap
            } else {
                renderer.register_bitmap_with_usage(
                    frame,
                    ruffle_render::backend::BitmapUsage::Excluded,
                )?
            };
            stream.bitmap = Some(handle.clone());

            Ok(BitmapInfo {
                handle,
                width,
                height,
            })
        })
    }
}

/// A single preloaded video stream.
pub struct VideoStream {
    bitmap: Option<BitmapHandle>,
    decoder: Box<dyn VideoDecoder>,
}

impl VideoStream {
    fn new(decoder: Box<dyn VideoDecoder>) -> Self {
        Self {
            decoder,
            bitmap: None,
        }
    }
}

#[cfg(all(test, feature = "screenvideo"))]
mod tests {
    use super::*;
    use ruffle_memory::MemoryManager;

    #[test]
    fn independent_stream_reference_keeps_decoder_until_last_release() {
        let manager = MemoryManager::new(16 << 20);
        let mut backend = SoftwareVideoBackend::new();
        let player = manager.player();
        let stream = player
            .enter(|| {
                backend.register_video_stream(
                    1,
                    (16, 16),
                    VideoCodec::ScreenVideo,
                    VideoDeblocking::UseVideoPacketValue,
                )
            })
            .unwrap();
        let script = stream.clone();
        drop((player, stream));
        assert_eq!(manager.snapshot().players.len(), 1);
        drop(script);
        assert!(manager.snapshot().players.is_empty());
    }
}
