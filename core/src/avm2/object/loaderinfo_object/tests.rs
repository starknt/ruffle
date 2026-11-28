use super::*;
use crate::avm2::globals::flash::display::loader::loader_allocator;
use crate::avm2::globals::slots::flash_display_loader as loader_slots;
use crate::display_object::MovieClip;
use crate::vminterface::Instantiator;

fn as3_player() -> crate::player::PlayerSession {
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
    crate::PlayerBuilder::new().with_movie(movie).build()
}

#[test]
fn unloading_avm1_content_from_an_as3_loader_stops_its_execution_list() {
    let player = as3_player();
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

#[test]
fn loader_unload_retires_display_uploads_and_preserves_exported_handles() {
    use crate::bitmap::{bitmap_data::BitmapData, operations};
    use crate::display_object::Bitmap;
    use std::sync::Arc;

    let session = as3_player();
    session
        .lock()
        .unwrap()
        .mutate_with_update_context(|context| {
            let mut activation = Activation::from_nothing(context);
            let loader =
                loader_allocator(activation.avm2().classes().loader, &mut activation).unwrap();
            let info = loader
                .get_slot(loader_slots::_CONTENT_LOADER_INFO)
                .as_object()
                .unwrap()
                .as_loader_info_object()
                .unwrap();
            let context = activation.context;
            let movie = *context.root_swf;
            let clip = MovieClip::new(movie, context.gc());
            let display = BitmapData::new(context.gc(), 4, 4, true, 0xff113355);
            operations::set_pixel32(context.gc(), context.renderer, display, 1, 1, 0xff668899)
                .unwrap();
            let shared = display.clone_data(context.gc(), context.renderer).unwrap();
            for (depth, data) in [display, shared].into_iter().enumerate() {
                let child = Bitmap::new_with_bitmap_data(context.gc(), 0, data, false, movie);
                clip.replace_at_depth(context, child.into(), depth as i32 + 1);
            }
            let raw = display.sync(context.renderer).unwrap();
            let upload = Arc::downgrade(
                &raw.borrow_mut(context.gc())
                    .update_dirty_texture(context.renderer)
                    .unwrap()
                    .0,
            );
            let exported = shared
                .bitmap_handle(context.gc(), context.renderer)
                .unwrap();
            let mut container = loader
                .as_stage_object()
                .unwrap()
                .display_object()
                .as_container()
                .unwrap();
            container.insert_at_index(context, clip.into(), 0);

            info.unload(context);
            assert!(
                upload.upgrade().is_none(),
                "unloaded subtree still pins its display upload"
            );
            assert_eq!(
                operations::get_pixel32(display, context.renderer, 1, 1).unwrap(),
                0xff668899
            );
            container.insert_at_index(context, clip.into(), 0);
            raw.borrow_mut(context.gc())
                .update_dirty_texture(context.renderer)
                .unwrap();
            assert_eq!(
                exported,
                shared
                    .bitmap_handle(context.gc(), context.renderer)
                    .unwrap()
            );
        });
}

#[test]
fn detached_clones_visit_each_bitmap_data_once() {
    use crate::bitmap::bitmap_data::BitmapData;
    use crate::display_object::Bitmap;
    let session = as3_player();
    session
        .lock()
        .unwrap()
        .mutate_with_update_context(|context| {
            let movie = *context.root_swf;
            let clip = MovieClip::new(movie, context.gc());
            let atlas = BitmapData::new(context.gc(), 16, 16, true, 0xff123456);
            let other = atlas.clone_data(context.gc(), context.renderer).unwrap();
            for depth in 0..128 {
                let data = if depth % 16 == 0 { other } else { atlas };
                let child = Bitmap::new_with_bitmap_data(context.gc(), 0, data, false, movie);
                clip.replace_at_depth(context, child.into(), depth);
            }
            let mut counts = [0, 0];
            for_each_detached_bitmap(clip.into(), |data| {
                counts[usize::from(data.ptr_eq(other))] += 1
            });
            assert_eq!(counts, [1, 1]);
        });
}
