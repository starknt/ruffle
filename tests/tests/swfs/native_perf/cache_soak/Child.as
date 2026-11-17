package {
    import flash.display.*;
    import flash.events.*;

    public class Child extends Sprite {
        private var receiver:Object = new Counter();
        private var checksum:int;
        public function Child() {
            var group:Sprite = new Sprite();
            addChild(group);
            for (var i:int = 0; i < 80; i++) {
                var s:Sprite = box(i);
                group.addChild(s);
            }
            var blend:Sprite = box(85);
            blend.blendMode = BlendMode.MULTIPLY;
            addChild(blend);
            var source:Sprite = box(86);
            var mask:Sprite = box(86);
            mask.x += 4;
            mask.alpha = 0.5;
            source.cacheAsBitmap = mask.cacheAsBitmap = true;
            addChild(source);
            addChild(mask);
            source.mask = mask;
            addEventListener(Event.ENTER_FRAME, step);
        }
        private function step(event:Event):void {
            for (var i:int = 0; i < 1000; i++) {
                receiver.value = (receiver.value + 1) & 255;
                checksum += receiver.step();
            }
        }
        private function box(i:int):Sprite {
            var s:Sprite = new Sprite();
            s.graphics.beginFill(0x48a0dd, 0.7);
            s.graphics.drawRect(0, 0, 12, 12);
            s.graphics.endFill();
            s.x = 4 + (i % 40) * 15;
            s.y = 4 + int(i / 40) * 15;
            return s;
        }
    }
}
class Counter {
    public var value:int;
    public function step():int { return value; }
}
