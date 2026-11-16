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

#[cfg(not(target_family = "wasm"))]
#[test]
fn worker_returns_shared_bytes_before_main_thread_installation() {
    let (_, movie) = futures::executor::block_on(prepare_swf_movie(
        movie_bytes(),
        "https://example.test/shared.swf".into(),
        None,
        None,
    ))
    .unwrap();
    assert!(
        movie.unwrap().shared_file().is_some(),
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

#[cfg(not(target_family = "wasm"))]
#[test]
fn worker_returns_inner_swf_without_requiring_main_thread_swz_extraction() {
    let bytes = movie_bytes();
    let (transport, movie) = futures::executor::block_on(prepare_swf_movie(
        swz(&bytes),
        "https://example.test/envelope.swz".into(),
        None,
        None,
    ))
    .unwrap();
    assert!(movie.is_some());
    assert!(
        transport.is_empty(),
        "parsed SWF must release transport bytes"
    );
    assert_eq!(movie.unwrap().compressed_len(), swz(&bytes).len());
}

#[cfg(not(target_family = "wasm"))]
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
    assert!(futures::executor::block_on(future).unwrap().1.is_some());
}

#[cfg(not(target_family = "wasm"))]
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
        let (worker_body, worker) = futures::executor::block_on(prepare_swf_movie(
            body.clone(),
            url.clone(),
            origin.clone(),
            info,
        ))
        .unwrap();
        let (inline_body, inline) = prepare_swf_movie_inline(body, url, origin, info).unwrap();
        let (worker, inline) = (worker.unwrap(), inline.unwrap());
        assert_eq!(worker_body, inline_body);
        assert!(worker_body.is_empty());
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

#[cfg(not(target_family = "wasm"))]
#[test]
fn worker_preserves_non_movie_payload_and_rejects_malformed_swf() {
    let image = b"GIF89a fixture".to_vec();
    let (body, movie) = futures::executor::block_on(prepare_swf_movie(
        image.clone(),
        "https://example.test/image.gif".into(),
        None,
        None,
    ))
    .unwrap();
    assert_eq!(body, image);
    assert!(movie.is_none());
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
