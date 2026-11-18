use crate::{PlayerBuilder, limits::ExecutionLimit, tag_utils::SwfMovieData};
use ruffle_render::profile::{RenderMetrics, enter};

#[test]
fn cached_descriptors_preserve_conversion_accessors_bound_methods_and_errors() {
    use std::{cell::RefCell, rc::Rc};
    struct Trace(Rc<RefCell<Vec<String>>>);
    impl crate::backend::log::LogBackend for Trace {
        fn avm_trace(&self, message: &str) {
            self.0.borrow_mut().push(message.into());
        }
        fn avm_warning(&self, _: &str) {}
    }
    let log = Rc::new(RefCell::new(Vec::new()));
    let movie = SwfMovieData::from_data(
        include_bytes!("../../../tests/tests/swfs/avm2/sealed_call_site_cache/test.swf"),
        "file:///pic-semantics.swf".into(),
        None,
        None,
    )
    .unwrap();
    let player = PlayerBuilder::new()
        .with_movie(movie)
        .with_log(Trace(log.clone()))
        .with_autoplay(true)
        .build();
    let metrics = RenderMetrics::default();
    let _profile = enter(Some(&metrics));
    let mut player = player.lock().unwrap();
    assert!(player.preload(&mut ExecutionLimit::none()));
    player.run_frame();
    assert_eq!(
        log.borrow().join("\n"),
        include_str!("../../../tests/tests/swfs/avm2/sealed_call_site_cache/output.txt").trim_end()
    );
    assert!(metrics.snapshot().work.pic_hits > 40);
}

#[test]
fn sealed_polymorphic_sites_hit_and_fifth_receiver_goes_generic() {
    for (bytes, mega) in [
        (
            &include_bytes!("../../../tests/tests/swfs/native_perf/traits_4/test.swf")[..],
            false,
        ),
        (
            &include_bytes!("../../../tests/tests/swfs/native_perf/traits_5/test.swf")[..],
            true,
        ),
    ] {
        let metrics = RenderMetrics::default();
        let _profile = enter(Some(&metrics));
        let movie = SwfMovieData::from_data(bytes, "file:///pic.swf".into(), None, None).unwrap();
        let player = PlayerBuilder::new()
            .with_movie(movie)
            .with_autoplay(true)
            .build();
        let mut player = player.lock().unwrap();
        assert!(player.preload(&mut ExecutionLimit::none()));
        for _ in 0..3 {
            player.run_frame();
        }
        let work = metrics.snapshot().work;
        if mega {
            assert!(work.pic_megamorphic > 0);
        } else {
            assert!(work.pic_hits > 1000, "{work:?}");
        }
    }
}
