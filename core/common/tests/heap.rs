use gc_arena::{Arena, Gc, GcWeak, Rootable};

#[test]
fn arena_bytes_remain_charged_until_a_weak_allocation_is_freed() {
    let mut arena = Arena::<Rootable![Option<GcWeak<'_, [u8; 256]>>]>::new(|_| None);
    let metrics = arena.metrics().clone();
    let initial = metrics.total_gc_bytes();
    arena.mutate_root(|mc, root| *root = Some(Gc::downgrade(Gc::new(mc, [0; 256]))));
    let allocated = metrics.total_gc_bytes();
    assert!(allocated >= initial + 256);
    arena.finish_cycle();
    assert_eq!(metrics.total_gc_bytes(), allocated);
    arena.mutate_root(|_, root| *root = None);
    arena.finish_cycle();
    assert_eq!(metrics.total_gc_bytes(), initial);
}
