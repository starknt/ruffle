use super::*;

fn runtime(limit: u64) -> GpuRuntime {
    GpuRuntime::new(GpuBudgetConfig {
        hard_limit: limit,
        soft_limit: limit / 4 * 3,
        ..GpuBudgetConfig::for_adapter(wgpu::DeviceType::DiscreteGpu, false)
    })
    .unwrap()
}

fn owner(id: u64) -> ResourceOwner {
    ResourceOwner::new(OwnerKind::Movie, id)
}

#[test]
fn pending_submission_bytes_charge_overlapping_uses_once_until_the_last_serial() {
    let gpu = runtime(16);
    let shared = allocation(&gpu, 8);
    let old = allocation(&gpu, 4);
    let bytes = || gpu.diagnostic_snapshot().in_flight_bytes;
    let first = gpu.register_submission([&shared, &old]).unwrap();
    let last = gpu.register_submission([&shared]).unwrap();
    drop(old);
    assert_eq!(bytes(), 12);
    assert!(gpu.complete_submission(first));
    assert_eq!(bytes(), 8);
    assert!(gpu.complete_submission(last));
    assert_eq!(bytes(), 0);
    assert_eq!(gpu.stats().active_bytes, 8);
}

fn allocation(gpu: &GpuRuntime, bytes: u64) -> ResourceLease {
    gpu.reserve(ResourceKind::Texture, owner(1), bytes)
        .unwrap()
        .commit(&[])
        .unwrap()
}

#[test]
fn outstanding_reservation_prevents_overcommit_and_drop_rolls_back() {
    let gpu = runtime(16);
    let reserved = gpu.reserve(ResourceKind::Texture, owner(1), 12).unwrap();
    assert_eq!(gpu.stats().reserved_bytes, 12);
    assert_eq!(gpu.stats().total_bytes, 12);
    assert_eq!(
        gpu.reserve(ResourceKind::Buffer, owner(2), 5)
            .unwrap_err()
            .kind(),
        RenderFailureKind::BudgetExceeded
    );
    assert_eq!(gpu.stats().total_bytes, 12);
    drop(reserved);
    assert_eq!(gpu.stats().total_bytes, 0);
    let _whole_budget = allocation(&gpu, 16);
    assert_eq!(gpu.stats().peak_bytes, 16);
}

#[test]
fn clones_keep_storage_charged_until_last_owner_is_dropped() {
    let gpu = runtime(16);
    let first = allocation(&gpu, 8);
    let second = first.clone();
    assert_eq!(first.id(), second.id());
    drop(first);
    assert_eq!(gpu.stats().active_bytes, 8);
    assert_eq!(gpu.stats().resources, 1);
    drop(second);
    assert_eq!(gpu.stats().total_bytes, 0);
    assert_eq!(gpu.stats().resources, 0);
}

#[test]
fn retired_storage_stays_charged_until_its_submission_completes() {
    let gpu = runtime(16);
    let first = allocation(&gpu, 12);
    let serial = gpu
        .register_submission(std::slice::from_ref(&first))
        .unwrap();
    drop(first);
    assert_eq!(gpu.stats().active_bytes, 0);
    assert_eq!(gpu.stats().retiring_bytes, 12);
    assert!(gpu.reserve(ResourceKind::Texture, owner(1), 5).is_err());
    assert!(gpu.complete_submission(serial));
    assert_eq!(gpu.stats().total_bytes, 0);
    assert!(!gpu.complete_submission(serial));
}

#[test]
fn completing_an_earlier_submission_does_not_release_a_later_use() {
    let gpu = runtime(16);
    let texture = allocation(&gpu, 12);
    let first = gpu
        .register_submission(std::slice::from_ref(&texture))
        .unwrap();
    let second = gpu
        .register_submission(std::slice::from_ref(&texture))
        .unwrap();
    drop(texture);
    gpu.complete_submission(first);
    assert_eq!(gpu.stats().retiring_bytes, 12);
    gpu.complete_submission(second);
    assert_eq!(gpu.stats().retiring_bytes, 0);
}

#[test]
fn idle_storage_remains_unavailable_to_other_submissions_until_completed() {
    let gpu = runtime(16);
    let texture = allocation(&gpu, 8);
    let serial = gpu
        .register_submission(std::slice::from_ref(&texture))
        .unwrap();
    texture.set_idle(true);
    assert_eq!(gpu.stats().active_bytes, 0);
    assert_eq!(gpu.stats().idle_bytes, 8);
    assert!(!texture.is_reusable());
    gpu.complete_submission(serial);
    assert!(texture.is_reusable());
    texture.set_idle(false);
    assert_eq!(gpu.stats().active_bytes, 8);
    assert_eq!(gpu.stats().idle_bytes, 0);
}

#[test]
fn bindings_keep_dependencies_alive_without_charging_texture_storage_twice() {
    let gpu = runtime(16);
    let texture = allocation(&gpu, 12);
    let view = gpu
        .reserve(ResourceKind::TextureView, owner(2), 0)
        .unwrap()
        .commit(std::slice::from_ref(&texture))
        .unwrap();
    let binding = gpu
        .reserve(ResourceKind::BindGroup, owner(2), 0)
        .unwrap()
        .commit(std::slice::from_ref(&view))
        .unwrap();
    drop(texture);
    drop(view);
    assert_eq!(gpu.stats().total_bytes, 12);
    assert_eq!(gpu.owner_bytes(owner(1)), 12);
    assert_eq!(gpu.owner_bytes(owner(2)), 0);
    let serial = gpu
        .register_submission(std::slice::from_ref(&binding))
        .unwrap();
    drop(binding);
    assert_eq!(gpu.stats().retiring_bytes, 12);
    assert_eq!(gpu.stats().resources, 3);
    gpu.complete_submission(serial);
    assert_eq!(gpu.stats().resources, 0);
}

#[test]
fn resources_from_another_device_cannot_be_bound_or_submitted() {
    let first = runtime(16);
    let second = runtime(16);
    let texture = allocation(&first, 8);
    assert_ne!(texture.id(), allocation(&second, 8).id());
    let result = second
        .reserve(ResourceKind::TextureView, owner(1), 0)
        .unwrap()
        .commit(std::slice::from_ref(&texture));
    assert_eq!(result.unwrap_err().kind(), RenderFailureKind::Validation);
    assert!(second.register_submission(&[texture]).is_err());
    assert_eq!(second.stats().reserved_bytes, 0);
}

#[test]
fn unregistered_or_foreign_completion_cannot_refund_live_work() {
    let first = runtime(16);
    let second = runtime(16);
    let a = allocation(&first, 8);
    let b = allocation(&second, 8);
    let serial = first.register_submission(&[a]).unwrap();
    let foreign = second.register_submission(&[b]).unwrap();
    assert!(!first.complete_submission(foreign));
    assert_eq!(first.stats().retiring_bytes, 8);
    assert!(first.complete_submission(serial));
}

#[test]
fn staging_limit_counts_reserved_and_retiring_storage() {
    let gpu = GpuRuntime::new(GpuBudgetConfig {
        hard_limit: 64,
        soft_limit: 48,
        staging_limit: 8,
        ..GpuBudgetConfig::for_adapter(wgpu::DeviceType::DiscreteGpu, false)
    })
    .unwrap();
    let pending = gpu
        .reserve(ResourceKind::StagingBuffer, owner(1), 8)
        .unwrap();
    assert!(
        gpu.reserve(ResourceKind::StagingBuffer, owner(1), 1)
            .is_err()
    );
    let staging = pending.commit(&[]).unwrap();
    let serial = gpu.register_submission(&[staging]).unwrap();
    assert!(
        gpu.reserve(ResourceKind::StagingBuffer, owner(1), 1)
            .is_err()
    );
    let _other_kind = allocation(&gpu, 16);
    gpu.complete_submission(serial);
    assert!(
        gpu.reserve(ResourceKind::StagingBuffer, owner(1), 8)
            .is_ok()
    );
}

#[test]
fn device_tiers_enforce_distinct_total_budgets_and_android_takes_precedence() {
    for (adapter, android, limit) in [
        (wgpu::DeviceType::DiscreteGpu, false, 2_147_483_648),
        (wgpu::DeviceType::IntegratedGpu, false, 1_073_741_824),
        (wgpu::DeviceType::Cpu, false, 1_073_741_824),
        (wgpu::DeviceType::DiscreteGpu, true, 536_870_912),
    ] {
        let gpu = GpuRuntime::new(GpuBudgetConfig::for_adapter(adapter, android)).unwrap();
        let _full = allocation(&gpu, limit);
        assert!(gpu.reserve(ResourceKind::Texture, owner(1), 1).is_err());
    }
}

#[test]
fn pressure_is_recomputed_after_reclamation() {
    let gpu = runtime(16);
    let a = allocation(&gpu, 12);
    assert_eq!(gpu.stats().pressure, MemoryPressure::Soft);
    let b = allocation(&gpu, 4);
    assert_eq!(gpu.stats().pressure, MemoryPressure::Hard);
    drop(a);
    assert_eq!(gpu.stats().pressure, MemoryPressure::Normal);
    drop(b);
}

#[test]
fn fatal_failure_blocks_new_work_and_keeps_the_first_cause() {
    let gpu = runtime(16);
    let pending = gpu.reserve(ResourceKind::Texture, owner(1), 8).unwrap();
    assert!(gpu.fail(RenderFailure::new(RenderFailureKind::DeviceLost, "lost")));
    assert!(!gpu.fail(RenderFailure::new(RenderFailureKind::Validation, "later")));
    assert_eq!(gpu.failure().unwrap().kind(), RenderFailureKind::DeviceLost);
    assert_eq!(
        pending.commit(&[]).unwrap_err().kind(),
        RenderFailureKind::DeviceLost
    );
    assert_eq!(gpu.stats().total_bytes, 0);
    assert!(gpu.reserve(ResourceKind::Texture, owner(1), 1).is_err());
    assert!(gpu.register_submission(&[]).is_err());
}

#[test]
fn byte_arithmetic_rejects_overflow_without_changing_ledger() {
    let gpu = runtime(u64::MAX);
    let a = allocation(&gpu, u64::MAX - 1);
    assert!(gpu.reserve(ResourceKind::Texture, owner(1), 8).is_err());
    assert_eq!(gpu.stats().total_bytes, u64::MAX - 1);
    drop(a);
    assert_eq!(gpu.stats().total_bytes, 0);
}

fn texture_descriptor() -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 7,
            height: 5,
            depth_or_array_layers: 3,
        },
        mip_level_count: 3,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bc1RgbaUnorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }
}

#[test]
fn texture_storage_includes_blocks_mips_layers_and_samples() {
    // BC1: (2*2 + 1*1 + 1*1) blocks * 8 bytes * 3 array layers.
    assert_eq!(texture_storage_bytes(&texture_descriptor()).unwrap(), 144);
    let desc = wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: 5,
            height: 3,
            depth_or_array_layers: 2,
        },
        mip_level_count: 1,
        sample_count: 4,
        format: wgpu::TextureFormat::Rgba8Unorm,
        ..texture_descriptor()
    };
    assert_eq!(texture_storage_bytes(&desc).unwrap(), 480);
    let volume = wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 4,
        },
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        ..texture_descriptor()
    };
    assert_eq!(texture_storage_bytes(&volume).unwrap(), 292);
}

#[test]
fn impossible_texture_sizes_are_rejected_before_resource_creation() {
    for size in [
        wgpu::Extent3d {
            width: 0,
            height: 1,
            depth_or_array_layers: 1,
        },
        wgpu::Extent3d {
            width: u32::MAX,
            height: u32::MAX,
            depth_or_array_layers: u32::MAX,
        },
    ] {
        assert!(
            texture_storage_bytes(&wgpu::TextureDescriptor {
                size,
                mip_level_count: 1,
                format: wgpu::TextureFormat::Rgba32Float,
                ..texture_descriptor()
            })
            .is_err()
        );
    }
}

#[test]
fn device_issues_distinct_owner_ids_for_recreated_movies() {
    let runtime = runtime(4096);
    let first = runtime.issue_owner(OwnerKind::Movie).unwrap();
    let next = runtime.issue_owner(OwnerKind::Movie).unwrap();
    assert_ne!(first, next);
    let lease = runtime
        .reserve(ResourceKind::Buffer, first, 512)
        .unwrap()
        .commit(&[])
        .unwrap();
    assert_eq!(runtime.owner_bytes(first), 512);
    assert_eq!(runtime.owner_bytes(next), 0);
    drop(lease);
    assert_eq!(runtime.owner_bytes(first), 0);
}

#[test]
fn third_pending_submission_is_rejected_until_the_oldest_completes() {
    let gpu = runtime(1024);
    let first = gpu.register_submission(&[]).unwrap();
    gpu.register_submission(&[]).unwrap();
    assert_eq!(
        gpu.register_submission(&[]).unwrap_err().kind(),
        RenderFailureKind::BudgetExceeded
    );
    assert!(gpu.complete_submission(first));
    assert!(gpu.register_submission(&[]).is_ok());
}
