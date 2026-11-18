use super::*;
use crate::avm2::globals::flash::display::loader::loader_allocator;
use crate::avm2::globals::slots::flash_display_loader as loader_slots;
use crate::display_object::MovieClip;
use crate::vminterface::Instantiator;

#[test]
fn unloading_avm1_content_from_an_as3_loader_stops_its_execution_list() {
    let mut bytes = Vec::new();
    swf::write_swf(
        &swf::Header::default_with_swf_version(10),
        &[
            swf::Tag::FileAttributes(swf::FileAttributes::IS_ACTION_SCRIPT_3),
            swf::Tag::ShowFrame,
        ],
        &mut bytes,
    )
    .unwrap();
    let movie = SwfMovieData::from_data(&bytes, "file:///root.swf".into(), None, None).unwrap();
    let player = crate::PlayerBuilder::new().with_movie(movie).build();
    let mut player = player.lock().unwrap();
    let weak: gc_arena::DynamicRoot<
        gc_arena::Rootable![(
            crate::tag_utils::SwfMovieWeak<'_>,
            crate::display_object::MovieClipWeak<'_>
        )],
    > = player.mutate_with_update_context(|context| {
        let mut activation = Activation::from_nothing(context);
        let loader = loader_allocator(activation.avm2().classes().loader, &mut activation).unwrap();
        let loader_info = loader
            .get_slot(loader_slots::_CONTENT_LOADER_INFO)
            .as_object()
            .unwrap()
            .as_loader_info_object()
            .unwrap();
        let context = activation.context;
        let movie = SwfMovie::new(context.gc(), SwfMovieData::empty(8, None));
        let content = MovieClip::new(movie, context.gc());
        let child = MovieClip::new(movie, context.gc());
        context
            .library
            .library_for_movie_mut(movie, context.gc_context);
        content.post_instantiation(context, None, Instantiator::Movie, false);
        child.post_instantiation(context, None, Instantiator::Movie, false);
        content.replace_at_depth(context, child.into(), 1);
        let mut container = loader
            .as_stage_object()
            .unwrap()
            .display_object()
            .as_container()
            .unwrap();
        assert!(container.raw_container().is_action_script_3());
        container.insert_at_index(context, content.into(), 0);
        loader_info.set_loader_stream(LoaderStream::Swf(movie, content.into()), context.gc());

        loader_info.unload(context);
        assert!(
            content.avm1_removed(),
            "unloading from an AS3 parent must stop the AVM1 movie"
        );
        assert!(child.avm1_removed(), "descendant clips must stop too");
        assert!(container.is_empty());
        crate::avm1::Avm1::run_frame(context);
        assert!(content.next_avm1_clip().is_none());
        assert!(child.next_avm1_clip().is_none());
        context.dynamic_root.stash(
            context.gc(),
            Gc::new(context.gc(), (movie.downgrade(), content.downgrade())),
        )
    });
    player.update(|context| {
        let metrics = context.gc().metrics();
        metrics.adjust_debt(metrics.total_gc_count() as f64 * 4.0 + 1024.0);
    });
    player.mutate_with_update_context(|context| {
        let (movie, content) = *context.dynamic_root.fetch(&weak);
        assert!(movie.upgrade(context.gc()).is_none());
        assert!(content.upgrade(context.gc()).is_none());
    });
}
