package {
    import flash.display.*;
    import flash.events.Event;
    import flash.filters.ColorMatrixFilter;
    import flash.geom.Rectangle;

    [SWF(width="64", height="24", frameRate="30", backgroundColor="#ffffff")]
    public class Test extends Sprite {
        private var group:Sprite = new Sprite();
        private var red:Sprite = box(0xff0000);
        private var blue:Sprite = box(0x0000ff);
        private var maskParent:Sprite = new Sprite();
        private var data:BitmapData = new BitmapData(8, 8, false, 0x00ff00);
        private var filtered:Sprite = box(0xff0000);
        private var frame:int;

        public function Test() {
            group.addChild(red);
            group.addChild(blue);
            for (var i:int = 0; i < 12; ++i) group.addChild(new Sprite());
            addChild(group);
            var masked:Sprite = box(0xff00ff);
            masked.x = 16;
            addChild(masked);
            var maskShape:Sprite = box(0xffffff);
            maskShape.scaleX = 0.5;
            maskParent.x = 16;
            maskParent.addChild(maskShape);
            addChild(maskParent);
            masked.mask = maskShape;
            filtered.x = 32;
            filtered.filters = [new ColorMatrixFilter([
                0,0,0,0,0, 1,0,0,0,0, 0,0,0,0,0, 0,0,0,1,0])];
            addChild(filtered);
            var bitmap:Bitmap = new Bitmap(data);
            bitmap.x = 48;
            addChild(bitmap);
            for (i = 0; i < 8; ++i) addChild(new Sprite());
            addEventListener(Event.ENTER_FRAME, tick);
        }

        private static function box(color:uint):Sprite {
            var s:Sprite = new Sprite();
            s.graphics.beginFill(color);
            s.graphics.drawRect(0, 0, 8, 8);
            s.graphics.endFill();
            return s;
        }

        private function sample():String {
            var b:BitmapData = new BitmapData(8, 8, false, 0);
            b.draw(group);
            return b.getPixel(2, 2).toString(16);
        }

        private function tick(e:Event):void {
            if (++frame != 3) return;
            blue.addEventListener(Event.REMOVED, function(e:Event):void {
                trace("removing", group.numChildren, group.getChildAt(1) == blue, group.getChildIndex(blue));
                group.setChildIndex(red, 2);
                trace("reordered", group.getChildAt(0) == blue, group.getChildIndex(red), sample());
            });
            group.removeChildAt(1);
            trace("removed", group.numChildren, group.getChildAt(1) == red, sample());
            group.addChildAt(blue, 0);
            trace("added", group.numChildren, group.getChildIndex(blue), group.getChildIndex(red), sample());
            trace("before add", group.width);
            var extent:Sprite = box(0xff0000);
            extent.x = 44;
            extent.addEventListener(Event.ADDED, function(e:Event):void {
                trace("during add", group.getChildIndex(extent), group.width);
            });
            group.addChild(extent);
            group.removeChild(extent);
            var from:Sprite = new Sprite();
            var to:Sprite = new Sprite();
            var moving:Sprite = new Sprite();
            from.addChild(moving);
            to.addChild(new Sprite());
            moving.addEventListener(Event.REMOVED, function(e:Event):void {
                to.removeChildAt(0);
            });
            to.addChild(moving);
            trace("reparented", from.numChildren, to.numChildren, to.getChildAt(0) == moving);
            group.scrollRect = new Rectangle(0, 0, 4, 8);
            maskParent.x += 4;
            filtered.filters = [];
            data.fillRect(data.rect, 0xffff00);
        }
    }
}
