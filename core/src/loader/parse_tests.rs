use super::*;
use crate::player::PlayerBuilder;

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
fn headless_session_fetches_sibling_resources_from_the_installed_root() {
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/tests/swfs/avm1/asnew/test.swf")
        .canonicalize()
        .unwrap();
    let mut session = PlayerBuilder::new().build();
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
    let session = PlayerBuilder::new().build_session();
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
    let mut session = PlayerBuilder::new().build_session();
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
    let player = PlayerBuilder::new()
        .with_movie(SwfMovieData::empty(8, None))
        .build();
    player.lock().unwrap().mutate_with_update_context(|uc| {
        let clip = uc.stage.root_clip().unwrap();
        let handle = uc.load_manager.add_loader(MovieLoader {
            self_handle: None,
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
            &[],
            transport.len(),
            "https://example.test/prepared.swf".into(),
            200,
            false,
            None,
            Some(prepared),
            1,
            0,
        );
        assert!(result.is_ok(), "prepared movie was reparsed: {result:?}");
        assert_eq!(clip.movie().data(), expected);
    });
}

#[test]
fn stale_load_bytes_parse_error_does_not_fail_a_new_loader_generation() {
    let player = PlayerBuilder::new()
        .with_movie(SwfMovieData::empty(8, None))
        .build();
    player.lock().unwrap().mutate_with_update_context(|uc| {
        let clip = uc.stage.root_clip().unwrap();
        let handle = uc.load_manager.add_loader(MovieLoader {
            self_handle: None,
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
        let inline = prepare_swf_movie_data(body, url, origin, info).unwrap();
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
    assert!(matches!(content, PreparedContent::Other(body) if body == image));
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
