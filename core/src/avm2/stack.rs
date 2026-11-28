//! Internal representation of the AVM2 value stack

use crate::avm2::method::Method;
use crate::avm2::value::Value;

use gc_arena::collect::Trace;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_memory::{MemoryBuffer, MemoryKind, MemoryScope};
use std::cell::{Cell, OnceCell};

const INITIAL_STACK_SIZE: usize = 4096;
const MAX_STACK_SIZE: usize = 200000;

/// The global, preallocated value stack. This has little use directly due to
/// the way AVM2 works; to do anything with it, an StackFrame must be
/// obtained first through the `get_stack_slice` method.
///
/// We use this instead of a `Vec<Value>` to allow for obtaining "mutable" (via
/// interior mutability) references to the stack without requiring a mutable
/// borrow on the stack or the `Avm2`.
#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct Stack<'gc>(Gc<'gc, StackData<'gc>>);

struct StackData<'gc> {
    /// Stack data
    stack: MemoryBuffer<Cell<Value<'gc>>>,
    overflow: OnceCell<MemoryBuffer<Cell<Value<'gc>>>>,

    /// The number of values currently stored on the stack
    stack_pointer: Cell<usize>,
}

impl<'gc> Stack<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        Stack(Gc::new(
            mc,
            StackData {
                stack: Self::allocate(INITIAL_STACK_SIZE),
                overflow: OnceCell::new(),
                stack_pointer: Cell::new(0),
            },
        ))
    }

    fn allocate(count: usize) -> MemoryBuffer<Cell<Value<'gc>>> {
        MemoryBuffer::new(
            &MemoryScope::current(),
            MemoryKind::AvmStack,
            vec![Cell::new(Value::Undefined); count],
        )
    }

    /// Returns a slice of stack data for the specified method
    pub fn get_frame_for_method(&self, method: Method<'gc>) -> StackFrame<'_, 'gc> {
        // Calculate the frame size
        let body = method
            .body()
            .expect("Cannot execute non-native method without body");
        let frame_size = body.max_stack as usize + body.num_locals as usize;

        self.get_frame(frame_size)
    }

    /// Returns a slice of stack data with the specified size, starting at the
    /// current stack pointer. Stack frames obtained from this method must be
    /// properly disposed of by using the `dispose_stack_frame` method.
    pub fn get_frame(&self, frame_size: usize) -> StackFrame<'_, 'gc> {
        // Create the stack frame
        let stack_pointer = &self.0.stack_pointer;

        let end = stack_pointer.get() + frame_size;
        let stack_data = if end <= INITIAL_STACK_SIZE {
            &self.0.stack
        } else {
            self.0
                .overflow
                .get_or_init(|| Self::allocate(MAX_STACK_SIZE))
        };
        let subslice = &stack_data[stack_pointer.get()..end];

        stack_pointer.set(stack_pointer.get() + frame_size);

        // Ensure the StackFrame returned to the caller does not have any
        // old values on it, as these may contain Gc pointers that have already
        // been collected.
        for value in subslice {
            value.set(Value::Undefined);
        }

        StackFrame::for_data(subslice)
    }

    pub fn dispose_stack_frame(self, stack_frame: StackFrame<'_, 'gc>) {
        self.0
            .stack_pointer
            .set(self.0.stack_pointer.get() - stack_frame.data.len());
    }
}

unsafe impl<'gc> Collect<'gc> for StackData<'gc> {
    // SAFETY: The only way to access values on the stack is by using StackFrames
    // obtained from get_stack_frame. StackFrame doesn't implement Collect, so
    // StackFrames from before a collection can't be accessed after the collection.
    fn trace<C: Trace<'gc>>(&self, _cc: &mut C) {
        // There should be no values on the value stack when collection is triggered
        assert_eq!(self.stack_pointer.get(), 0);
    }
}

/// A stack frame for a particular method. Despite its name, this stores both
/// method locals and method stack.
pub struct StackFrame<'a, 'gc> {
    data: &'a [Cell<Value<'gc>>],
    stack_pointer: Cell<usize>,
}

impl<'a, 'gc> StackFrame<'a, 'gc> {
    pub fn empty() -> StackFrame<'a, 'gc> {
        Self {
            data: &[],
            stack_pointer: Cell::new(0),
        }
    }

    fn for_data(data: &'a [Cell<Value<'gc>>]) -> StackFrame<'a, 'gc> {
        Self {
            data,
            stack_pointer: Cell::new(0),
        }
    }

    /// Push a value onto the operand stack.
    #[inline(always)]
    pub fn push(&self, value: Value<'gc>) {
        self.data[self.stack_pointer.get()].set(value);
        self.stack_pointer.set(self.stack_pointer.get() + 1);
    }

    /// Retrieve the top-most value on the operand stack.
    #[inline(always)]
    pub fn pop(&self) -> Value<'gc> {
        self.stack_pointer.set(self.stack_pointer.get() - 1);

        self.data[self.stack_pointer.get()].get()
    }

    /// Peek the n-th value from the end of the operand stack.
    #[inline(always)]
    pub fn peek(&self, index: usize) -> Value<'gc> {
        self.data[self.stack_pointer.get() - index - 1].get()
    }

    #[inline(always)]
    pub fn stack_top(&self) -> &'a Cell<Value<'gc>> {
        &self.data[self.stack_pointer.get() - 1]
    }

    #[inline(always)]
    pub fn value_at(&self, index: usize) -> Value<'gc> {
        self.data[index].get()
    }

    #[inline(always)]
    pub fn set_value_at(&self, index: usize, value: Value<'gc>) {
        self.data[index].set(value);
    }

    #[inline(always)]
    pub fn pop_args(&self, arg_count: u32) -> &'a [Cell<Value<'gc>>] {
        let arg_count = arg_count as usize;
        let base = self.stack_pointer.get() - arg_count;

        self.set_stack_pointer(base);

        &self.data[base..base + arg_count]
    }

    #[inline(always)]
    pub fn set_stack_pointer(&self, size: usize) {
        self.stack_pointer.set(size);
    }

    /// Get the number of entries currently on the stack.
    pub fn len(&self) -> usize {
        self.stack_pointer.get()
    }

    /// Move out of this StackFrame, leaving it empty and creating a new StackFrame
    /// that points to the same data as this one.
    pub fn take(&mut self) -> Self {
        let new_frame = Self {
            data: self.data,
            stack_pointer: self.stack_pointer.clone(),
        };

        self.data = &[];
        self.stack_pointer = Cell::new(0);

        new_frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frames_across_the_initial_stack_keep_their_values_and_clear_on_reuse() {
        gc_arena::arena::rootless_mutate(|mc| {
            let stack = Stack::new(mc);
            let outer = stack.get_frame(INITIAL_STACK_SIZE - 2);
            outer.push(42.into());
            let deep = stack.get_frame(8);
            deep.push(73.into());
            assert_eq!(outer.peek(0).as_i32(), 42);
            assert_eq!(deep.pop().as_i32(), 73);
            stack.dispose_stack_frame(deep);
            let next = stack.get_frame(8);
            assert!(matches!(next.value_at(0), Value::Undefined));
            stack.dispose_stack_frame(next);
            assert_eq!(outer.pop().as_i32(), 42);
            stack.dispose_stack_frame(outer);
        });
    }
}
