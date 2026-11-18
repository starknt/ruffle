package {
    import flash.display.*;
    import flash.events.*;

    [SWF(width="130", height="126", frameRate="60", backgroundColor="#ffffff")]
    public class Test extends Sprite {
        private var pixels:BitmapData = new BitmapData(130, 126, false, 0);
        private var source:Shape = new Shape();
        private var frame:int = 0;

        public function Test() {
            addChild(new Bitmap(pixels));
            addEventListener(Event.ENTER_FRAME, step);
        }

        private function step(event:Event):void {
            frame++;
            var base:uint = 0x123400 + (frame % 127);
            source.graphics.clear();
            for (var quadrant:int = 0; quadrant < 4; quadrant++) {
                source.graphics.beginFill(base ^ (quadrant * 0x303030));
                source.graphics.drawRect((quadrant % 2) * 65, int(quadrant / 2) * 63, 65, 63);
                source.graphics.endFill();
            }
            pixels.draw(source);
            var valid:Boolean = true;
            for (var y:int = 0; y < 126; y++) {
                for (var x:int = 0; x < 130; x++) {
                    var q:int = (x < 65 ? 0 : 1) + (y < 63 ? 0 : 2);
                    var expected:uint = 0xff000000 | (base ^ (q * 0x303030));
                    if (pixels.getPixel32(x, y) != expected) valid = false;
                }
            }
            if (frame <= 3 || !valid) trace("scan", frame, valid);
        }
    }
}
