package {
    import flash.display.Bitmap;
    import flash.display.BitmapData;
    import flash.display.Sprite;
    import flash.events.Event;

    [SWF(width="400", height="180", frameRate="60", backgroundColor="#17202b")]
    public class Test extends Sprite {
        private var row:Sprite = new Sprite();
        private var tick:int;

        public function Test() {
            addChild(row);
            for (var i:int = 0; i < 384; ++i) {
                var color:uint = 0xb0000000 | ((i * 0x15431) & 0xffffff);
                var bitmap:Bitmap = new Bitmap(new BitmapData(16, 16, true, color));
                bitmap.smoothing = true;
                bitmap.x = i % 32 * 11;
                bitmap.y = int(i / 32) * 11;
                row.addChild(bitmap);
            }
            addEventListener(Event.ENTER_FRAME, advance);
        }

        private function advance(event:Event):void {
            ++tick;
            row.x = 0.25 + tick % 17;
            row.y = 0.125 + tick % 11;
        }
    }
}
