use ruffle_core::{
    PlayerBuilder,
    memory::{MemoryKind, MemoryManager},
};

#[test]
fn players_share_the_host_budget_but_release_their_own_arena_bytes() {
    let manager = MemoryManager::new(8 << 20);
    let make = || {
        PlayerBuilder::new()
            .with_memory_manager(manager.clone())
            .build()
    };
    let a = make();
    let b = make();
    let snapshot = manager.snapshot();
    assert_eq!(snapshot.players.len(), 2);
    for player in snapshot.players {
        assert!(player.active[&MemoryKind::GcHeap] > 0);
    }
    drop(a);
    assert_eq!(manager.snapshot().players.len(), 1);
    drop(b);
    assert!(manager.snapshot().players.is_empty());
}
