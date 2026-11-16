//! Published AVM2 instructions are immutable after verification and optimization.
use super::op::Op;
use gc_arena::collect::Trace;
use gc_arena::{Collect, Gc, GcWeak};

pub struct VerifiedCode<'gc> {
    ops: Vec<Op<'gc>>,
    roots: Option<Box<[Gc<'gc, ()>]>>,
}

impl<'gc> VerifiedCode<'gc> {
    pub fn new(mut ops: Vec<Op<'gc>>) -> Self {
        let input_bytes = std::mem::size_of_val(&ops) + ops.capacity() * std::mem::size_of::<Op>();
        ops.shrink_to_fit();
        let mut roots = RootIndex::default();
        ops.trace(&mut roots);
        let roots = roots.finish();
        let code = Self { ops, roots };
        ruffle_render::profile::code_published(
            input_bytes,
            code.storage_bytes(),
            code.ops.len(),
            code.roots
                .as_ref()
                .map_or(code.ops.len(), |roots| roots.len()),
        );
        code
    }

    pub fn as_slice(&self) -> &[Op<'gc>] {
        &self.ops
    }

    fn storage_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.ops.capacity() * std::mem::size_of::<Op>()
            + self
                .roots
                .as_ref()
                .map_or(0, |roots| std::mem::size_of_val(&**roots))
    }
}

// SAFETY: The instruction vector is private and only exposes a shared slice.
// Op is Copy: its GC edges cannot be replaced through inline interior mutability.
// LookupSwitch's integer offset rewrites do not change these direct edges.
// Children inside GC allocations (e.g. Class/Method) may still change through
// their existing write barriers: the arena traces those children dynamically
// using the allocation's original Collect metadata, not this direct-edge index.
// Build the index using Op's own Collect implementation after all verifier/
// optimizer rewrites. It contains every unique strong edge, without changing
// ownership; weak edges force direct tracing of the original vector.
// This type has no Drop implementation that accesses GC pointers.
unsafe impl<'gc> Collect<'gc> for VerifiedCode<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        if let Some(roots) = &self.roots {
            roots.trace(cc);
        } else {
            self.ops.trace(cc);
        }
    }
}

#[derive(Default)]
struct RootIndex<'gc> {
    strong: Vec<Gc<'gc, ()>>,
    has_weak: bool,
}

impl<'gc> Trace<'gc> for RootIndex<'gc> {
    fn trace_gc(&mut self, gc: Gc<'gc, ()>) {
        self.strong.push(gc);
    }

    fn trace_gc_weak(&mut self, _: GcWeak<'gc, ()>) {
        self.has_weak = true;
    }
}

impl<'gc> RootIndex<'gc> {
    fn finish(mut self) -> Option<Box<[Gc<'gc, ()>]>> {
        if self.has_weak {
            return None;
        }
        self.strong
            .sort_unstable_by_key(|gc| Gc::as_ptr(*gc) as usize);
        self.strong.dedup_by(|a, b| Gc::ptr_eq(*a, *b));
        Some(self.strong.into_boxed_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avm2::op::LookupSwitch;
    use gc_arena::collect::Trace;
    use gc_arena::{Arena, Gc, GcWeak, Rootable};
    use std::{cell::Cell, collections::HashSet, mem::size_of};

    #[derive(Default)]
    struct Edges {
        visits: usize,
        ids: HashSet<usize>,
    }

    impl<'gc> Trace<'gc> for Edges {
        fn trace_gc(&mut self, gc: Gc<'gc, ()>) {
            self.visits += 1;
            self.ids.insert(Gc::as_ptr(gc) as usize);
        }

        fn trace_gc_weak(&mut self, _: GcWeak<'gc, ()>) {
            panic!("this fixture has no weak references");
        }
    }

    #[test]
    fn published_code_releases_growth_capacity_without_changing_instructions() {
        let mut ops = Vec::with_capacity(4096);
        for i in 0..512 {
            ops.push(Op::PushInt { value: i });
        }
        let baseline_bytes = size_of::<Vec<Op>>() + ops.capacity() * size_of::<Op>();
        let code = VerifiedCode::new(ops);
        assert_eq!(code.as_slice().len(), 512);
        for (i, op) in code.as_slice().iter().enumerate() {
            assert!(matches!(op, Op::PushInt { value } if *value == i as i32));
        }
        let bytes = size_of::<VerifiedCode>()
            + code.ops.capacity() * size_of::<Op>()
            + code
                .roots
                .as_ref()
                .map_or(0, |r| std::mem::size_of_val(&**r));
        assert!(
            bytes < baseline_bytes / 2,
            "published code kept growth capacity: {bytes} bytes"
        );
    }

    #[test]
    fn published_code_traces_each_unique_reference_once() {
        Arena::<Rootable![()]>::new(|mc| {
            let switch = Gc::new(
                mc,
                LookupSwitch {
                    default_offset: Cell::new(3),
                    case_offsets: vec![Cell::new(7)].into_boxed_slice(),
                },
            );
            let ops = vec![Op::LookupSwitch(switch); 1024];
            let optimized = VerifiedCode::new(ops);
            let mut indexed = Edges::default();
            optimized.trace(&mut indexed);
            assert_eq!(
                indexed.visits, 1,
                "repeated opcode GC references must be deduplicated"
            );
        });
    }

    #[test]
    fn indexed_code_keeps_switch_payload_alive_until_code_is_released() {
        let mut arena =
            Arena::<Rootable![(Option<VerifiedCode<'_>>, GcWeak<'_, LookupSwitch>)]>::new(|mc| {
                let switch = Gc::new(
                    mc,
                    LookupSwitch {
                        default_offset: Cell::new(11),
                        case_offsets: vec![Cell::new(7), Cell::new(19)].into_boxed_slice(),
                    },
                );
                let code = VerifiedCode::new(vec![Op::LookupSwitch(switch); 32]);
                (Some(code), Gc::downgrade(switch))
            });
        arena.finish_cycle();
        arena.mutate(|mc, (code, weak)| {
            assert!(
                weak.upgrade(mc).is_some(),
                "indexed strong edge was collected"
            );
            let Op::LookupSwitch(switch) = code.as_ref().unwrap().as_slice()[0] else {
                panic!()
            };
            assert_eq!(switch.default_offset.get(), 11);
            assert_eq!(switch.case_offsets[1].get(), 19);
            switch.default_offset.set(5);
        });
        arena.finish_cycle();
        arena.mutate_root(|_, (code, _)| {
            let Op::LookupSwitch(switch) = code.as_ref().unwrap().as_slice()[0] else {
                panic!()
            };
            assert_eq!(
                switch.default_offset.get(),
                5,
                "integer payload mutation changed GC roots"
            );
            *code = None;
        });
        arena.finish_cycle();
        arena.mutate(|mc, (_, weak)| {
            assert!(
                weak.upgrade(mc).is_none(),
                "index kept an unreachable method payload alive"
            );
        });
    }

    #[test]
    fn indexed_erased_roots_trace_namespace_string_children() {
        use crate::avm2::Namespace;
        use crate::string::{AvmString, AvmStringInterner, StringContext};
        let mut arena =
            Arena::<Rootable![(AvmStringInterner<'_>, Option<VerifiedCode<'_>>)]>::new(|mc| {
                (AvmStringInterner::new(mc), None)
            });
        arena.mutate_root(|mc, (interner, code)| {
            let mut context = StringContext::from_parts(mc, interner);
            let name = AvmString::new_utf8(mc, String::from("owned-method-namespace"));
            let namespace = Namespace::internal(name, &mut context);
            *code = Some(VerifiedCode::new(vec![Op::PushNamespace { namespace }; 32]));
        });
        arena.finish_cycle();
        arena.mutate(|_, (_, code)| {
            let Op::PushNamespace { namespace } = code.as_ref().unwrap().as_slice()[0] else {
                panic!()
            };
            assert_eq!(
                namespace.as_uri_opt().unwrap().to_utf8_lossy(),
                "owned-method-namespace"
            );
        });
    }

    #[test]
    fn weak_references_force_direct_instruction_tracing() {
        Arena::<Rootable![()]>::new(|mc| {
            let value = Gc::new(mc, 7);
            let mut roots = RootIndex::default();
            value.trace(&mut roots);
            Gc::downgrade(value).trace(&mut roots);
            assert!(
                roots.finish().is_none(),
                "weak references cannot be promoted to strong roots"
            );
        });
    }

    #[test]
    #[ignore = "optional construction/GC benchmark; emits measurements without timing thresholds"]
    fn benchmark_published_code_construction_and_gc() {
        use std::time::Instant;
        let mut construction = Vec::new();
        let mut collection = Vec::new();
        let mut bytes = 0;
        for round in 0..34 {
            let start = Instant::now();
            let mut arena = Arena::<Rootable![Vec<VerifiedCode<'_>>]>::new(|mc| {
                let switch = Gc::new(
                    mc,
                    LookupSwitch {
                        default_offset: Cell::new(7),
                        case_offsets: vec![Cell::new(3), Cell::new(5)].into_boxed_slice(),
                    },
                );
                (0..512)
                    .map(|_| {
                        let mut ops = Vec::with_capacity(1024);
                        for i in 0..512 {
                            ops.push(if i % 8 == 0 {
                                Op::LookupSwitch(switch)
                            } else {
                                Op::PushInt { value: i }
                            });
                        }
                        VerifiedCode::new(ops)
                    })
                    .collect()
            });
            let built = start.elapsed().as_secs_f64() * 1000.0;
            arena.mutate(|_, code| {
                bytes = code.iter().map(VerifiedCode::storage_bytes).sum::<usize>()
            });
            let start = Instant::now();
            arena.finish_cycle();
            let collected = start.elapsed().as_secs_f64() * 1000.0;
            if round >= 3 {
                construction.push(built);
                collection.push(collected);
            }
        }
        construction.sort_by(f64::total_cmp);
        collection.sort_by(f64::total_cmp);
        println!(
            "verifiedCodeBenchmark optimized=true samples=31 storageBytes={} constructP50Ms={:.3} constructP95Ms={:.3} gcP50Ms={:.3} gcP95Ms={:.3}",
            bytes, construction[15], construction[29], collection[15], collection[29]
        );
    }
}
