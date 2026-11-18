package {
    import flash.display.Sprite;
    public class ParentCounter extends Sprite {
        private var count:int;
        public function get value():int { return count; }
        public function set value(n:int):void { count = n; }
        public function add(n:int):int { return count + n; }
    }
}
