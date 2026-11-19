package {
    import flash.display.Sprite;
    import flash.events.Event;

    [SWF(width="1280", height="720", frameRate="60", backgroundColor="#ffffff")]
    public class Test extends Sprite {
        private var items:Array = [];
        private var frame:int = 0;

        public function Test() {
            var group:Sprite = this;
            for (var i:int = 0; i < 2000; ++i) {
                if (i % 40 == 0) { group = new Sprite(); addChild(group); }
                var box:Sprite = new Sprite();
                box.graphics.beginFill(0x48a0dd, 0.7);
                box.graphics.drawRect(0, 0, 12, 12);
                box.x = 4 + (i % 80) * 15;
                box.y = 4 + int(i / 80) * 15;
                group.addChild(box);
                items.push(box);
            }
            addEventListener(Event.ENTER_FRAME, step);
        }

        private function step(event:Event):void {
            ++frame;
            for (var i:int = 0; i < 20; ++i) {
                items[(frame + i * 97) % items.length].x += frame % 2 ? 1 : -1;
            }
        }
    }
}
