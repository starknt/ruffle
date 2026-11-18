package {
    import flash.display.*;
    import flash.events.*;
    [SWF(width="800", height="600", frameRate="60", backgroundColor="#ffffff")]
    public class Test extends Sprite {
        private var shapes:Array = [];
        private var frame:int = 0;
        public function Test() {
            for (var i:int = 0; i < 320; i++) {
                var s:Shape = new Shape(); s.x = (i % 32) * 24 + 4; s.y = int(i / 32) * 48 + 4;
                addChild(s); shapes.push(s);
            }
            addEventListener(Event.ENTER_FRAME, step);
        }
        private function step(event:Event):void {
            frame++;
            var wobble:Number = (frame % 17) / 17;
            for (var i:int = 0; i < shapes.length; i++) {
                var g:Graphics = shapes[i].graphics; g.clear();
                g.beginFill(0x4070c0 + (i % 16) * 0x402, 0.8);
                g.moveTo(1 + wobble, 0);
                g.curveTo(14, -2, 20, 7 + wobble);
                g.curveTo(27, 14, 18, 23);
                g.curveTo(25, 35, 10, 39);
                g.curveTo(-3, 32, 2, 24);
                g.curveTo(9, 19, 0, 13);
                g.curveTo(-4, 4, 1 + wobble, 0);
                g.endFill();
            }
        }
    }
}
