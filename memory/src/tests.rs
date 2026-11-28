use super::*;
use std::sync::Arc;

#[test]
fn pending_and_shared_bytes_follow_the_last_owner_across_players() {
    let manager = MemoryManager::new(1024);
    let first = manager.player();
    let second = manager.player();
    let pending = first.try_reserve(MemoryKind::Decode, 768).unwrap();
    assert!(second.try_reserve(MemoryKind::Decode, 257).is_none());
    let charge = Arc::new(pending.commit(512));
    let view = charge.clone();
    let cancelled = second.try_reserve(MemoryKind::Decode, 512).unwrap();
    assert_eq!(manager.snapshot().reserved_bytes, 512);
    assert_eq!(manager.snapshot().active_bytes, 512);
    drop((cancelled, charge, first));
    assert_eq!(manager.snapshot().active_bytes, 512);
    assert_eq!(manager.snapshot().reserved_bytes, 0);
    drop(view);
    assert_eq!(manager.snapshot().active_bytes, 0);
    assert_eq!(manager.snapshot().players.len(), 1);
    drop(second);
    assert!(manager.snapshot().players.is_empty());
}

#[test]
fn pressure_relaxes_only_after_crossing_the_hysteresis_boundary() {
    let mut state = PressureState::default();
    let mut update = |used| state.update(used, 100, PressureLevel::Normal);
    assert_eq!(update(64), PressureLevel::Normal);
    assert_eq!(update(65), PressureLevel::Moderate);
    assert_eq!(update(80), PressureLevel::High);
    assert_eq!(update(95), PressureLevel::Critical);
    assert_eq!(update(91), PressureLevel::Critical);
    assert_eq!(update(89), PressureLevel::High);
    assert_eq!(update(76), PressureLevel::High);
    assert_eq!(update(74), PressureLevel::Moderate);
    assert_eq!(update(59), PressureLevel::Normal);
}

#[test]
fn two_players_share_one_emergency_gc_per_process_pressure_cycle() {
    let manager = MemoryManager::new(100);
    let a = manager.player();
    let b = manager.player();
    let mut used = a.track(MemoryKind::Audio, 98);
    assert!(a.gc_pressure().2);
    assert!(!b.gc_pressure().2);
    used.resize(89);
    assert_eq!(b.gc_pressure().1.level, PressureLevel::High);
    used.resize(97);
    assert!(b.gc_pressure().2);
    assert!(!a.gc_pressure().2);
}

#[test]
fn optional_work_accounts_for_uncommitted_reservations_above_rss_and_os_pressure() {
    let manager = MemoryManager::new(1024);
    let player = manager.player();
    manager.sample_rss(Some(800));
    let held = player.try_reserve(MemoryKind::Network, 160).unwrap();
    assert!(player.try_reserve(MemoryKind::Network, 96).is_none());
    assert_eq!(manager.pressure().level, PressureLevel::High);
    drop(held);
    manager.notify_pressure(PressureLevel::Critical);
    assert!(player.try_reserve(MemoryKind::Network, 1).is_none());
}
