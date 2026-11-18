use super::*;
use crate::avm2::{Namespace, QName};
use crate::binary_data::BinaryData;
use crate::tag_utils::{SwfMovieData, SwfMovieWeak};
use gc_arena::{Arena, Gc, GcWeak, Rootable};

#[derive(Collect)]
#[collect(no_drop)]
struct Roots<'gc> {
    library: Library<'gc>,
    movie: Option<SwfMovie<'gc>>,
    class: Option<Avm2Class<'gc>>,
    weak_movie: SwfMovieWeak<'gc>,
    weak_symbol: GcWeak<'gc, BinaryData>,
}

fn roots<'gc>(mc: &Mutation<'gc>) -> Roots<'gc> {
    let movie = SwfMovie::new(mc, SwfMovieData::empty(10, None));
    let symbol = Gc::new(
        mc,
        BinaryData::from_swf_tag(movie, &swf::DefineBinaryData { id: 7, data: &[] }),
    );
    let mut library = Library::empty();
    library
        .library_for_movie_mut(movie, mc)
        .register_character(7, Character::BinaryData(symbol));
    Roots {
        library,
        movie: Some(movie),
        class: None,
        weak_movie: movie.downgrade(),
        weak_symbol: Gc::downgrade(symbol),
    }
}

#[test]
fn movie_index_releases_unreachable_library_symbols_after_collection() {
    let mut arena = Arena::<Rootable![Roots<'_>]>::new(roots);
    arena.finish_cycle();
    arena.mutate(|mc, root| {
        let movie = root.movie.unwrap();
        let symbol = root
            .library
            .library_for_movie(movie, mc)
            .unwrap()
            .character_by_id(7);
        assert!(matches!(symbol, Some(Character::BinaryData(_))));
        assert!(root.weak_symbol.upgrade(mc).is_some());
    });
    arena.mutate_root(|_, root| root.movie = None);
    arena.finish_cycle();
    arena.mutate(|mc, root| {
        assert!(
            root.weak_movie.upgrade(mc).is_none(),
            "the global index must not root a movie"
        );
        assert!(
            root.weak_symbol.upgrade(mc).is_none(),
            "its library symbols must be released too"
        );
    });
}

#[test]
fn retained_symbol_class_keeps_its_movie_until_the_class_is_unreachable() {
    let mut arena = Arena::<Rootable![Roots<'_>]>::new(|mc| {
        let mut root = roots(mc);
        let class = Avm2Class::custom_new(
            QName::new(Namespace::any(), AvmString::new_utf8(mc, "RetainedSymbol")),
            None,
            None,
            Box::default(),
            mc,
        );
        root.library
            .avm2_class_registry_mut()
            .set_class_symbol(class, root.movie.unwrap(), 7, mc);
        // SymbolClass is first-wins, both within a movie and across movies.
        root.library
            .avm2_class_registry_mut()
            .set_class_symbol(class, root.movie.unwrap(), 9, mc);
        let other = SwfMovie::new(mc, SwfMovieData::empty(10, None));
        root.library
            .avm2_class_registry_mut()
            .set_class_symbol(class, other, 11, mc);
        root.class = Some(class);
        root.movie = None;
        root
    });
    arena.finish_cycle();
    arena.mutate(|mc, root| {
        let (movie, id) = root
            .library
            .avm2_class_registry()
            .class_symbol(root.class.unwrap())
            .unwrap();
        assert_eq!(id, 7);
        assert!(SwfMovie::ptr_eq(
            &movie,
            &root.weak_movie.upgrade(mc).unwrap()
        ));
        assert!(
            root.library
                .library_for_movie(movie, mc)
                .unwrap()
                .contains_character(7)
        );
        assert!(root.weak_symbol.upgrade(mc).is_some());
    });
    arena.mutate_root(|_, root| root.class = None);
    arena.finish_cycle();
    arena.mutate(|mc, root| {
        assert!(
            root.weak_movie.upgrade(mc).is_none(),
            "the class registry must not root a class/movie cycle"
        );
        assert!(root.weak_symbol.upgrade(mc).is_none());
    });
}
