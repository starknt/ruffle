use super::*;
use crate::player::PlayerBuilder;

fn fixture_player() -> PlayerBuilder {
    // A poll in this fixture must not wait for unrelated tests to return file-read permits.
    let budget = ruffle_memory::MemoryManager::process().budget_bytes();
    PlayerBuilder::new().with_memory_manager(ruffle_memory::MemoryManager::new(budget))
}

fn movie_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    swf::write_swf(
        &swf::Header::default_with_swf_version(8),
        &[swf::Tag::ShowFrame],
        &mut bytes,
    )
    .unwrap();
    bytes
}

#[test]
fn external_loader_image_uploads_its_actual_read_only_when_drawn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/tests/swfs/avm2/loader_image/test.swf")
        .canonicalize()
        .unwrap();
    let url = Url::from_file_path(&path).unwrap();
    let movie =
        SwfMovieData::from_data(&std::fs::read(path).unwrap(), url.into(), None, None).unwrap();
    let mut generations = Vec::new();
    for version in 1..=2 {
        let renderer = crate::bitmap::symbol_pixels::tests::renderer(false);
        let uploads = renderer.loader_uploads.clone();
        let mut session = fixture_player()
            .with_movie(movie.clone())
            .with_renderer(renderer)
            .build();
        session.lock().unwrap().run_frame();
        let request = session.lock().unwrap().mutate_with_update_context(|uc| {
            if version == 2 {
                let [first] = uc.load_manager.active_reads.as_slice() else {
                    panic!("expected one Loader.load")
                };
                let first_id = first.id;
                let loader = uc.load_manager.get_loader(first.movie.unwrap()).unwrap();
                let replacement = uc.load_movie_into_clip(
                    loader.target_clip,
                    Request::get("test.png".into()),
                    None,
                    loader.vm_data,
                );
                uc.spawn_future(replacement);
                assert_ne!(uc.load_manager.active_reads[0].id, first_id);
            }
            let [read] = uc.load_manager.active_reads.as_slice() else {
                panic!("expected one active Loader read")
            };
            LoaderRequest {
                session_generation: uc.load_manager.session_generation,
                request_id: read.id,
                resource_version: version,
            }
        });
        generations.push(request.session_generation);
        session.poll_tasks();
        assert!(
            uploads.borrow().is_empty(),
            "VM commit eagerly registered pixels"
        );
        session.lock().unwrap().render().unwrap();
        session.lock().unwrap().render().unwrap();
        assert_eq!(uploads.borrow().as_slice(), &[request]);
    }
    assert_ne!(generations[0], generations[1]);
}

#[test]
fn headless_session_fetches_sibling_resources_from_the_installed_root() {
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/tests/swfs/avm1/asnew/test.swf")
        .canonicalize()
        .unwrap();
    let mut session = fixture_player().build();
    session.lock().unwrap().fetch_root_movie(
        Url::from_file_path(&file).unwrap().into(),
        vec![],
        Box::new(|_| {}),
    );
    session.poll_tasks();
    let response = session
        .lock()
        .unwrap()
        .fetch(Request::get("output.txt".into()), FetchReason::UrlLoader);
    let body = futures::executor::block_on(async {
        response
            .await
            .map_err(|response| response.error)?
            .body()
            .await
    })
    .unwrap();
    assert_eq!(
        body,
        std::fs::read(file.with_file_name("output.txt")).unwrap()
    );
}

#[test]
fn a_replaced_root_load_cannot_deliver_metadata_or_install_its_movie() {
    use std::cell::RefCell;
    use std::rc::Rc;
    let session = fixture_player().build_session();
    let player = session.player();
    let seen = Rc::new(RefCell::new(Vec::new()));
    let request = |name: &'static str| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/tests/swfs/avm1/asnew/test.swf")
            .canonicalize()
            .unwrap();
        let url = Url::from_file_path(path).unwrap().to_string();
        let seen = seen.clone();
        player.lock().unwrap().mutate_with_update_context(|ctx| {
            load_root_movie(
                ctx,
                Request::get(url),
                vec![],
                Box::new(move |_| seen.borrow_mut().push(name)),
            )
        })
    };
    let first = request("old");
    let second = request("new");
    futures::executor::block_on(second).unwrap();
    futures::executor::block_on(first).unwrap();
    assert_eq!(*seen.borrow(), ["new"]);
}

#[test]
fn metadata_callback_can_replace_the_root_before_installation() {
    let mut session = fixture_player().build_session();
    let player = session.player().clone();
    let newer = player.clone();
    let swfs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/tests/swfs/avm1");
    let url = |name: &str| {
        Url::from_file_path(swfs.join(name).join("test.swf").canonicalize().unwrap())
            .unwrap()
            .to_string()
    };
    let first = url("asnew");
    let second = url("load_cancel_via_unloadmovie");
    let replacement = second.clone();
    let load = player.lock().unwrap().mutate_with_update_context(|uc| {
        load_root_movie(
            uc,
            Request::get(first.clone()),
            vec![],
            Box::new(move |_| {
                newer
                    .lock()
                    .unwrap()
                    .fetch_root_movie(replacement, vec![], Box::new(|_| {}));
            }),
        )
    });
    futures::executor::block_on(load).unwrap();
    let current_url = || {
        player
            .lock()
            .unwrap()
            .mutate_with_update_context(|ctx| ctx.root_swf.url().to_owned())
    };
    assert_ne!(current_url(), first);
    session.poll_tasks();
    assert_eq!(current_url(), second);
}

fn root_fixture(name: &str) -> String {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/tests/swfs/avm1");
    Url::from_file_path(dir.join(name).canonicalize().unwrap())
        .unwrap()
        .into()
}

fn file_session(url: &str) -> crate::PlayerSession {
    let mut movie = SwfMovieData::empty(8, None);
    movie.set_url(url.into());
    fixture_player().with_movie(movie).build()
}

fn avm1_root_request(player: &crate::PlayerSession, url: String) -> OwnedFuture<(), Error> {
    player.lock().unwrap().mutate_with_update_context(|uc| {
        let clip = uc.stage.root_clip().unwrap();
        uc.load_movie_into_clip(
            clip,
            Request::get(url),
            None,
            MovieLoaderVMData::Avm1 {
                broadcaster: None,
                base_clip: clip,
            },
        )
    })
}

#[test]
fn host_and_level_zero_replacements_only_deliver_the_latest_root() {
    for host_is_last in [true, false] {
        for newest_first in [true, false] {
            let initial = root_fixture("load_cancel_via_unloadmovie/child.swf");
            let session = file_session(&initial);
            let host_url = initial.clone();
            let clip_url = root_fixture("load_cancel_via_unloadmovie/test.swf");
            let host = || {
                session.lock().unwrap().mutate_with_update_context(|uc| {
                    load_root_movie(uc, Request::get(host_url.clone()), vec![], Box::new(|_| {}))
                })
            };
            let avm = || avm1_root_request(&session, clip_url.clone());
            let (old, latest) = if host_is_last {
                (avm(), host())
            } else {
                (host(), avm())
            };
            let first_url = if host_is_last { &host_url } else { &clip_url };
            let (first, second) = if newest_first {
                (latest, old)
            } else {
                (old, latest)
            };
            futures::executor::block_on(first).unwrap();
            futures::executor::block_on(second).unwrap();
            session
                .lock()
                .unwrap()
                .mutate_with_update_context(|uc| assert_eq!(uc.root_swf.url(), first_url));
        }
    }
}

#[test]
fn non_movie_level_zero_response_finishes_its_loader() {
    let url = root_fixture("asnew/output.txt");
    let session = file_session(&url);
    futures::executor::block_on(avm1_root_request(&session, url)).unwrap();
    session.lock().unwrap().mutate_with_update_context(|uc| {
        assert!(
            uc.load_manager
                .movies
                .values()
                .all(|l| l.loader_status == LoaderStatus::Failed)
        );
    });
}

#[test]
fn native_file_access_follows_the_selected_movies_directory() {
    let file = root_fixture("asnew/output.txt");
    for root in [
        root_fixture("load_cancel_via_unloadmovie/test.swf"),
        "https://example.test/movie.swf".into(),
    ] {
        let session = file_session(&root);
        let request = session
            .lock()
            .unwrap()
            .fetch(Request::get(file.clone()), FetchReason::UrlLoader);
        assert!(
            futures::executor::block_on(request).is_err(),
            "{root} unexpectedly opened a file outside its directory"
        );
    }
}

#[test]
fn prepared_movie_is_installed_without_reparsing_transport_bytes() {
    let bytes = movie_bytes();
    let prepared = SwfMovieData::from_data(
        &bytes,
        "https://example.test/prepared.swf".into(),
        None,
        None,
    )
    .unwrap();
    let expected = prepared.data().to_vec();
    let player = fixture_player()
        .with_movie(SwfMovieData::empty(8, None))
        .build();
    player.lock().unwrap().mutate_with_update_context(|uc| {
        let clip = uc.stage.root_clip().unwrap();
        let handle = uc.load_manager.add_loader(MovieLoader {
            self_handle: None,
            clip_read: None,
            generation: 1,
            target_clip: clip,
            vm_data: MovieLoaderVMData::Avm1 {
                broadcaster: None,
                base_clip: clip,
            },
            loader_status: LoaderStatus::Pending,
            movie: None,
        });
        // Poison the transport envelope. The successful worker result is
        // authoritative; any main-thread SWZ extraction must fail.
        let transport = b"not-a-valid-swz";
        let result = MovieLoader::movie_loader_data(
            handle,
            uc,
            transport.len(),
            "https://example.test/prepared.swf".into(),
            200,
            false,
            None,
            PreparedContent::Movie(prepared),
            1,
            0,
            None,
        );
        assert!(result.is_ok(), "prepared movie was reparsed: {result:?}");
        assert_eq!(clip.movie().data(), expected);
    });
}

#[test]
fn stale_load_bytes_parse_error_does_not_fail_a_new_loader_generation() {
    let player = fixture_player()
        .with_movie(SwfMovieData::empty(8, None))
        .build();
    player.lock().unwrap().mutate_with_update_context(|uc| {
        let clip = uc.stage.root_clip().unwrap();
        let handle = uc.load_manager.add_loader(MovieLoader {
            self_handle: None,
            clip_read: None,
            generation: 2,
            target_clip: clip,
            vm_data: MovieLoaderVMData::Avm1 {
                broadcaster: None,
                base_clip: clip,
            },
            loader_status: LoaderStatus::Pending,
            movie: None,
        });
        MovieLoader::movie_loader_parse_error(
            handle,
            uc,
            1,
            "https://example.test/stale.swf".into(),
        )
        .unwrap();
        assert_eq!(
            uc.load_manager.get_loader(handle).unwrap().loader_status,
            LoaderStatus::Pending
        );
    });
}

#[test]
fn worker_returns_shared_bytes_before_main_thread_installation() {
    let movie = futures::executor::block_on(prepare_swf_movie(
        movie_bytes(),
        "https://example.test/shared.swf".into(),
        None,
        None,
    ))
    .unwrap();
    assert!(
        movie.into_movie().unwrap().shared_file().is_some(),
        "shared-file preparation must finish on the parsing worker"
    );
}

fn swz(bytes: &[u8]) -> Vec<u8> {
    // Minimal ASN.1 container accepted by the SWZ reader; no signatures or
    // external assets are needed to exercise the envelope normalization.
    fn tlv(tag: u8, payload: &[u8]) -> Vec<u8> {
        assert!(payload.len() < 128);
        [vec![tag, payload.len() as u8], payload.to_vec()].concat()
    }
    let content = tlv(
        0x30,
        &[vec![0x05, 0], tlv(0xa0, &tlv(0x04, bytes))].concat(),
    );
    let signed = tlv(0x30, &[vec![0x05, 0, 0x05, 0], content].concat());
    tlv(0x30, &tlv(0xa0, &signed))
}

#[test]
fn worker_returns_inner_swf_without_requiring_main_thread_swz_extraction() {
    let bytes = movie_bytes();
    let movie = futures::executor::block_on(prepare_swf_movie(
        swz(&bytes),
        "https://example.test/envelope.swz".into(),
        None,
        None,
    ))
    .unwrap();
    assert_eq!(
        movie.into_movie().unwrap().compressed_len(),
        swz(&bytes).len()
    );
}

#[test]
fn malformed_swz_with_trailing_der_keeps_its_unknown_body() {
    let mut envelope = swz(&movie_bytes());
    envelope.push(0x30); // incomplete second ASN.1 block
    let result = prepare_swf_movie_data(envelope.clone(), String::new(), None, None, None).unwrap();
    assert!(matches!(result, PreparedContent::Other(body) if body == envelope));
}

#[test]
fn worker_parse_yields_before_work_can_run_on_a_single_thread_pool() {
    use std::future::Future;
    use std::task::Context;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let mut future = Box::pin(prepare_swf_movie(
        movie_bytes(),
        "https://example.test/queued.swf".into(),
        None,
        None,
    ));
    pool.install(|| {
        // The only worker is executing this poll, so queued parsing cannot
        // finish until we return. A synchronous fallback would return Ready.
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(future.as_mut().poll(&mut context).is_pending());
    });
    assert!(matches!(
        futures::executor::block_on(future).unwrap(),
        PreparedContent::Movie(_)
    ));
}

#[test]
fn inline_and_worker_preparation_preserve_bytes_metadata_and_shared_storage() {
    use ruffle_common::sandbox::SandboxType;
    for body in [movie_bytes(), swz(&movie_bytes())] {
        let compressed_len = body.len();
        let info = Some(LoadBytesInfo {
            loader_sandbox_type: SandboxType::LocalWithFile,
        });
        let url = "https://example.test/movie.swf?mode=fixture".to_string();
        let origin = Some("https://example.test/parent.swf".to_string());
        let worker = futures::executor::block_on(prepare_swf_movie(
            body.clone(),
            url.clone(),
            origin.clone(),
            info,
        ))
        .unwrap();
        let inline = prepare_swf_movie_data(body, url, origin, info, None).unwrap();
        let (worker, inline) = (worker.into_movie().unwrap(), inline.into_movie().unwrap());
        assert_eq!(worker.data(), inline.data());
        assert_eq!(worker.url(), inline.url());
        assert_eq!(worker.parameters(), inline.parameters());
        assert_eq!(worker.sandbox_type(), SandboxType::LocalWithFile);
        assert!(worker.shared_file().is_some());
        assert!(inline.shared_file().is_some());
        assert_eq!(worker.compressed_len(), compressed_len);
        assert_eq!(inline.compressed_len(), compressed_len);
    }
}

#[test]
fn worker_preserves_non_movie_payload_and_rejects_malformed_swf() {
    let image = b"GIF89a fixture".to_vec();
    let content = futures::executor::block_on(prepare_swf_movie(
        image.clone(),
        "https://example.test/image.gif".into(),
        None,
        None,
    ))
    .unwrap();
    assert!(matches!(content, PreparedContent::Image {source, pixels: Err(_)} if source == image));
    for bytes in [b"FWS\x08\0\0\0\0".to_vec(), swz(b"FWS\x08\0\0\0\0")] {
        assert!(
            futures::executor::block_on(prepare_swf_movie(
                bytes,
                "https://example.test/malformed.swf".into(),
                None,
                None,
            ))
            .is_err()
        );
    }
}

#[test]
#[ignore = "diagnostic parser cost; accepts AOLA_PARSE_BENCH_SWF, never asserts wall-clock thresholds"]
fn removed_main_thread_parse_cost_benchmark() {
    use std::hint::black_box;
    let body = std::env::var_os("AOLA_PARSE_BENCH_SWF")
        .map(|path| std::fs::read(path).unwrap())
        .unwrap_or_else(movie_bytes);
    let mut parse = Vec::new();
    let mut share = Vec::new();
    let mut expanded = 0;
    for _ in 0..31 {
        let started = Instant::now();
        let movie = SwfMovieData::from_data(
            black_box(&body),
            "https://example.test/benchmark.swf".into(),
            None,
            None,
        )
        .unwrap();
        parse.push(started.elapsed().as_nanos());
        expanded = movie.data().len();
        let started = Instant::now();
        let movie = movie.with_shared_bytes(true);
        share.push(started.elapsed().as_nanos());
        black_box(movie);
    }
    parse.sort_unstable();
    share.sort_unstable();
    println!(
        "compressed_bytes={} expanded_tag_bytes={} redundant_parse_ns_p50={} redundant_parse_ns_p95={} shared_prepare_ns_p50={} shared_prepare_ns_p95={}",
        body.len(),
        expanded,
        parse[15],
        parse[29],
        share[15],
        share[29],
    );
}

#[derive(Default, Clone)]
struct Trace(std::rc::Rc<std::cell::RefCell<Vec<String>>>);

impl crate::backend::log::LogBackend for Trace {
    fn avm_trace(&self, message: &str) {
        self.0.borrow_mut().push(message.to_string());
    }
    fn avm_warning(&self, _: &str) {}
}

#[test]
fn child_replacement_and_close_during_local_load_bytes_only_commit_the_replacement() {
    let trace = Trace::default();
    let renderer = crate::bitmap::symbol_pixels::tests::renderer(false);
    let uploads = renderer.loader_uploads.clone();
    let url = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/tests/swfs/avm2/loader_image/test.swf");
    let movie = SwfMovieData::from_data(
        include_bytes!("movie/tests/ChildCancellation.swf"),
        Url::from_file_path(url.canonicalize().unwrap())
            .unwrap()
            .into(),
        None,
        None,
    )
    .unwrap();
    let mut session = fixture_player()
        .with_movie(movie)
        .with_renderer(renderer)
        .with_log(trace.clone())
        .build();
    for _ in 0..4 {
        session.lock().unwrap().run_frame();
        session.poll_tasks();
    }
    assert_eq!(*trace.0.borrow(), ["replacement bytes 0", "complete 100"]);
    session.lock().unwrap().render().unwrap();
    let uploads = uploads.borrow();
    assert_eq!(uploads.len(), 1);
    assert_eq!(uploads[0].resource_version, 2);
    session.lock().unwrap().mutate_with_update_context(|uc| {
        use crate::avm2::globals::slots::flash_display_loader as loader_slots;
        assert_eq!(uc.load_manager.pending_counts(), (0, 0));
        let root = uc.stage.root_clip().unwrap().as_container().unwrap();
        let closed = root.child_by_index(1).unwrap().object2().unwrap();
        let info = closed
            .get_slot(loader_slots::_CONTENT_LOADER_INFO)
            .as_object()
            .unwrap()
            .as_loader_info_object()
            .unwrap();
        assert!(
            matches!(info.loader_stream(), LoaderStream::NotYetLoaded(_, None, _)),
            "closed LoaderInfo retained its pending bitmap"
        );
        assert!(info.loader_stream().movie().data().is_empty());
    });
}

#[test]
fn closing_a_remote_load_bytes_retires_both_superseded_preparations() {
    let trace = Trace::default();
    let movie = SwfMovieData::from_data(
        include_bytes!("../../../tests/tests/swfs/avm2/loader_reentrant_close/test.swf"),
        "https://example.test/loader.swf".into(),
        None,
        None,
    )
    .unwrap();
    let mut session = fixture_player()
        .with_movie(movie)
        .with_log(trace.clone())
        .build();
    session.lock().unwrap().run_frame();
    session
        .lock()
        .unwrap()
        .mutate_with_update_context(|uc| assert_eq!(uc.load_manager.pending_counts(), (0, 0)));
    session.poll_tasks();
    assert_eq!(&*trace.0.borrow(), &["closed byte loads"]);
}
