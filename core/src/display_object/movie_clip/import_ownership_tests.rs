use super::*;
use crate::backend::navigator::{
    ErrorResponse, FileNavigator, NavigationMethod, NavigatorBackend, OwnedFuture, SuccessResponse,
};
use crate::socket::{SocketActionSender, SocketHandle, SocketWrite};
use futures::channel::oneshot;
use std::path::Path;
use url::Url;

// Hold the first real file response so replacement happens after the import has started.
struct DelayedImport {
    files: FileNavigator,
    first: RefCell<Option<oneshot::Receiver<()>>>,
}

impl NavigatorBackend for DelayedImport {
    fn set_root_url(&mut self, url: &str) {
        self.files.set_root_url(url);
    }
    fn fetch(&self, request: Request) -> OwnedFuture<Box<dyn SuccessResponse>, ErrorResponse> {
        let response = self.files.fetch(request);
        let wait = self.first.take();
        Box::pin(async move {
            let file = response.await?;
            if let Some(wait) = wait {
                wait.await.unwrap();
            }
            Ok(file)
        })
    }
    fn resolve_url(&self, url: &str) -> Result<Url, url::ParseError> {
        self.files.resolve_url(url)
    }
    fn pre_process_url(&self, url: Url) -> Url {
        url
    }
    fn navigate_to_url(
        &self,
        _: &str,
        _: &str,
        _: Option<(NavigationMethod, indexmap::IndexMap<String, String>)>,
    ) {
        unreachable!()
    }
    fn connect_socket(
        &mut self,
        _: String,
        _: u16,
        _: std::time::Duration,
        _: SocketHandle,
        _: async_channel::Receiver<SocketWrite>,
        _: SocketActionSender,
    ) {
        unreachable!()
    }
}

fn swf(tags: &[swf::Tag<'_>], version: u8) -> Vec<u8> {
    let mut bytes = Vec::new();
    swf::write_swf(
        &swf::Header::default_with_swf_version(version),
        tags,
        &mut bytes,
    )
    .unwrap();
    bytes
}

fn importing_movie(path: &Path, library: &str, version: u8) -> SwfMovieData {
    let bytes = swf(
        &[
            swf::Tag::ImportAssets {
                url: swf::SwfStr::from_utf8_str(library),
                imports: vec![swf::ExportedAsset {
                    id: 42,
                    name: swf::SwfStr::from_utf8_str("symbol"),
                }],
            },
            swf::Tag::ShowFrame,
        ],
        version,
    );
    std::fs::write(path, &bytes).unwrap();
    SwfMovieData::from_data(
        &bytes,
        Url::from_file_path(path).unwrap().into(),
        None,
        None,
    )
    .unwrap()
}

fn exported_symbol<'gc>(uc: &UpdateContext<'gc>, movie: SwfMovie<'gc>) -> Option<String> {
    let library = uc.library.library_for_movie(movie, uc.gc())?;
    let Character::MovieClip(symbol) = library.character_by_id(42)? else {
        panic!("expected sprite")
    };
    Some(symbol.movie().url().into())
}

fn replace_during_import(replace_root: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.swf");
    let library = swf(
        &[
            swf::Tag::DefineSprite(swf::Sprite {
                id: 1,
                num_frames: 1,
                tags: vec![swf::Tag::ShowFrame],
            }),
            swf::Tag::ExportAssets(vec![swf::ExportedAsset {
                id: 1,
                name: swf::SwfStr::from_utf8_str("symbol"),
            }]),
            swf::Tag::ShowFrame,
        ],
        7,
    );
    for name in ["old.swf", "independent.swf", "new.swf"] {
        std::fs::write(dir.path().join(name), &library).unwrap();
    }
    let (release, wait) = oneshot::channel();
    let mut session = crate::PlayerBuilder::new()
        .with_movie(importing_movie(&path, "old.swf", 7))
        .with_navigator(DelayedImport {
            files: FileNavigator::default(),
            first: RefCell::new(Some(wait)),
        })
        .build_session();
    assert!(!session.lock().unwrap().preload(&mut ExecutionLimit::none()));
    session.poll_tasks();
    let (retired, independent) = session.lock().unwrap().mutate_with_update_context(|uc| {
        let old = uc.stage.root_clip().unwrap().as_movie_clip().unwrap();
        let old_movie = old.movie();
        if replace_root {
            uc.replace_root_movie(importing_movie(&path, "new.swf", 8));
        }
        let independent = {
            let movie = SwfMovie::new(uc.gc(), importing_movie(&path, "independent.swf", 7));
            let clip = MovieClip::new(movie, uc.gc());
            clip.replace_with_movie(uc, Some(movie), false, None);
            assert!(!clip.preload(uc, &mut ExecutionLimit::none()));
            MovieClipHandle::stash(uc, clip)
        };
        if !replace_root {
            let movie = SwfMovie::new(uc.gc(), importing_movie(&path, "new.swf", 8));
            old.replace_with_movie(uc, Some(movie), true, None);
        }
        let current = uc.stage.root_clip().unwrap().as_movie_clip().unwrap();
        assert!(!current.preload(uc, &mut ExecutionLimit::none()));
        (
            uc.dynamic_root
                .stash::<Rootable![SwfMovie<'_>]>(uc.gc(), Gc::new(uc.gc(), old_movie)),
            independent,
        )
    });
    session.poll_tasks();
    assert!(
        release.send(()).is_err(),
        "retired read still holds its response"
    );
    session.lock().unwrap().mutate_with_update_context(|uc| {
        let current = uc.stage.root_clip().unwrap().as_movie_clip().unwrap();
        assert_eq!(
            exported_symbol(uc, *uc.dynamic_root.fetch(&retired)),
            None,
            "retired import installed an export"
        );
        for (clip, file) in [
            (current, "new.swf"),
            (independent.fetch(uc), "independent.swf"),
        ] {
            assert!(clip.preload(uc, &mut ExecutionLimit::none()));
            assert_eq!(
                exported_symbol(uc, clip.movie()),
                Some(Url::from_file_path(dir.path().join(file)).unwrap().into())
            );
        }
    });
}

#[test]
fn root_replacement_during_import() {
    replace_during_import(true);
}

#[test]
fn clip_replacement_during_independent_imports() {
    replace_during_import(false);
}
