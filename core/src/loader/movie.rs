//! Preparation owns transport bytes. Only the finished movie or non-SWF body crosses back to the player.
use super::{ContentType, Error, Instant, LoadBytesInfo, SwfMovieData, extract_swz};

pub(super) enum PreparedContent {
    Movie(SwfMovieData),
    Other(Vec<u8>),
}

impl PreparedContent {
    pub(super) fn into_movie(self) -> Result<SwfMovieData, Error> {
        match self {
            Self::Movie(movie) => Ok(movie),
            Self::Other(bytes) => Err(Error::UnexpectedData(
                ContentType::Swf,
                ContentType::sniff(&bytes),
            )),
        }
    }
}

/// Prepare SWF bytes on the shared rayon pool without holding the player lock.
pub(super) async fn prepare_swf_movie(
    body: Vec<u8>,
    url: String,
    loader_url: Option<String>,
    load_bytes_info: Option<LoadBytesInfo>,
) -> Result<PreparedContent, swf::error::Error> {
    let (sender, receiver) = async_channel::bounded(1);
    rayon::spawn_fifo(move || {
        let result = prepare_swf_movie_data(body, url, loader_url, load_bytes_info);
        let _ = sender.send_blocking(result);
    });
    receiver
        .recv()
        .await
        .expect("SWF parser worker dropped result")
}

/// Fully prepare immutable movie data, including the SWZ envelope and shared
/// backing file. Callers install this result without parsing or copying it again.
pub(super) fn prepare_swf_movie_data(
    body: Vec<u8>,
    url: String,
    loader_url: Option<String>,
    load_bytes_info: Option<LoadBytesInfo>,
) -> Result<PreparedContent, swf::error::Error> {
    let transport = body;
    let uncompressed = if ContentType::sniff(&transport) == ContentType::Swf {
        None
    } else {
        extract_swz(&transport).ok()
    };
    let parse_body = uncompressed.as_deref().unwrap_or(&transport);
    if ContentType::sniff(parse_body) != ContentType::Swf {
        return Ok(PreparedContent::Other(transport));
    }
    let movie = SwfMovieData::from_data(parse_body, url, loader_url, load_bytes_info)?
        .with_shared_bytes(true)
        .with_compressed_len(transport.len());
    // The movie records the original transport length for progress/bytesTotal;
    // do not send the transport buffer back through the loader channel.
    Ok(PreparedContent::Movie(movie))
}

pub(super) fn use_inline_movie_parse(url: &str) -> bool {
    url.starts_with("file:") || url.starts_with("data:") || url.starts_with("ruffle:")
}

pub(super) async fn prepare_swf_movie_for_url(
    body: Vec<u8>,
    url: String,
    loader_url: Option<String>,
    load_bytes_info: Option<LoadBytesInfo>,
) -> (Result<PreparedContent, swf::error::Error>, u64) {
    let started = Instant::now();
    let result = if use_inline_movie_parse(&url) {
        prepare_swf_movie_data(body, url, loader_url, load_bytes_info)
    } else {
        prepare_swf_movie(body, url, loader_url, load_bytes_info).await
    };
    (result, started.elapsed().as_nanos() as u64)
}
