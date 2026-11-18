package {
    import flash.display.Sprite;
    public class Child extends ParentCounter {
        public function step(amount:int):int {
            super.value = amount;
            return super.add(1) + super.value;
        }
    }
}
