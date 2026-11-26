package {
    import flash.display.Shape;
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.filters.BlurFilter;
    import flash.filters.ColorMatrixFilter;

    [SWF(width="512", height="384", frameRate="60", backgroundColor="#28374b")]
    public class Test extends Sprite {
        private var tile:Shape = new Shape();
        private var tick:int = 0;

        public function Test() {
            tile.filters = [
                new BlurFilter(5.5, 3.5, 2),
                new ColorMatrixFilter([0,1,0,0,0, 0,0,1,0,0, 1,0,0,0,0, 0,0,0,1,0])
            ];
            addChild(tile);
            addEventListener(Event.ENTER_FRAME, draw);
            draw(null);
        }

        private function draw(event:Event):void {
            ++tick;
            tile.graphics.clear();
            tile.graphics.beginFill(0x407cb6, 0.55);
            tile.graphics.drawRoundRect(4, 4, 500, 372, 19, 19);
            tile.graphics.endFill();
            tile.graphics.beginFill(0xef7839, 0.8);
            tile.graphics.drawCircle(40 + tick % 430, 180, 29);
            tile.graphics.endFill();
        }
    }
}
