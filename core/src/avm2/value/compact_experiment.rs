//! Isolated layout trial. References stay in a GC-traced side table so pointer
//! provenance and the exact bits of colliding NaNs are preserved.
use super::Value;
use gc_arena::Collect;

const PREFIX: u64 = 0x7ffc_0000_0000_0000;
const MASK: u64 = 0xffff_0000_0000_0000;
const TAG_SHIFT: u32 = 40;
const PAYLOAD: u64 = (1 << TAG_SHIFT) - 1;

#[derive(Default, Collect)]
#[collect(no_drop)]
struct Packed<'gc> {
    words: Vec<u64>,
    side: Vec<Value<'gc>>,
}
impl<'gc> Packed<'gc> {
    fn push(&mut self, value: Value<'gc>) {
        let tagged = |tag: u64, bits: u64| PREFIX | (tag << TAG_SHIFT) | bits;
        self.words.push(match value {
            Value::Number(n) if n.to_bits() & MASK != PREFIX => n.to_bits(),
            Value::Integer(i) => tagged(1, u64::from(i as u32)),
            Value::Bool(b) => tagged(2, u64::from(b)),
            Value::Null => tagged(3, 0),
            Value::Undefined => tagged(4, 0),
            value => {
                let index = self.side.len() as u64;
                assert!(index <= PAYLOAD);
                self.side.push(value);
                tagged(0, index)
            }
        });
    }
    fn get(&self, index: usize) -> Value<'gc> {
        let word = self.words[index];
        if word & MASK != PREFIX {
            return Value::Number(f64::from_bits(word));
        }
        match (word & !MASK) >> TAG_SHIFT {
            0 => self.side[(word & PAYLOAD) as usize],
            1 => Value::Integer(word as u32 as i32),
            2 => Value::Bool(word & 1 != 0),
            3 => Value::Null,
            4 => Value::Undefined,
            _ => unreachable!(),
        }
    }
    fn bytes(&self) -> usize {
        self.words.capacity() * size_of::<u64>() + self.side.capacity() * size_of::<Value>()
    }
}

#[test]
fn numeric_and_reference_heavy_layouts_round_trip_across_collection() {
    use crate::string::AvmString;
    use gc_arena::{Arena, Rootable};
    let mut arena = Arena::<Rootable![Packed<'_>]>::new(|mc| {
        let mut packed = Packed::default();
        for _ in 0..128 {
            for value in [
                Value::Number(-0.0),
                Value::Number(f64::INFINITY),
                Value::Number(f64::from_bits(PREFIX | 42)),
                Value::Integer(i32::MIN),
                Value::Bool(true),
                Value::Null,
                Value::Undefined,
                Value::Number(0.5),
            ] {
                packed.push(value);
            }
        }
        let baseline = packed.words.len() * size_of::<Value>();
        println!(
            "numeric mix: {} bytes baseline, {} bytes packed",
            baseline,
            packed.bytes()
        );
        assert!(packed.bytes() < baseline);
        let string = AvmString::new_utf8(mc, "live reference in compact trial");
        let mut references = Packed::default();
        for _ in 0..1024 {
            references.push(Value::String(string));
        }
        println!(
            "reference mix: {} bytes baseline, {} bytes packed",
            baseline,
            references.bytes()
        );
        assert!(references.bytes() >= baseline);
        packed.push(Value::String(string));
        packed
    });
    arena.finish_cycle();
    arena.mutate(|_, packed| {
        assert_eq!(packed.get(0).as_f64().to_bits(), (-0.0_f64).to_bits());
        assert_eq!(packed.get(2).as_f64().to_bits(), PREFIX | 42);
        assert_eq!(packed.get(3).as_i32(), i32::MIN);
        assert!(matches!(packed.get(4), Value::Bool(true)));
        assert!(matches!(packed.get(5), Value::Null));
        assert!(matches!(packed.get(6), Value::Undefined));
        assert!(matches!(packed.get(1024), Value::String(s) if s.to_utf8_lossy() == "live reference in compact trial"));
    });
}
