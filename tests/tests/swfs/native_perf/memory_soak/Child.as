package {
    import flash.display.*;
    import flash.filters.BlurFilter;
    import flash.geom.Rectangle;
    import flash.text.TextField;
    import flash.utils.ByteArray;

    public class Child extends Sprite {
        private var source:BitmapData;
        private var copy:BitmapData;
        private var bytes:ByteArray = new ByteArray();
        private var values:Vector.<uint> = new Vector.<uint>(65536);
        public function Child() {
            source = new BitmapData(1024, 1024, true, 0xff337755);
            copy = source.clone();
            copy.lock();
            copy.fillRect(new Rectangle(0, 0, 256, 256), 0xff6584ab);
            copy.unlock();
            if (source.getPixel32(1, 1) != 0xff337755) throw new Error("clone corrupted source");
            bytes.length = 2 * 1024 * 1024;
            values[65535] = 0xffabcdef;
            if (values[65535] != 0xffabcdef) throw new Error("uint storage");
            var bg:Bitmap = new Bitmap(copy);
            bg.scaleX = bg.scaleY = 0.25;
            addChild(bg);
            for (var i:int=0; i<90; i++) {
                var shape:Shape = new Shape();
                shape.graphics.beginFill(0x245980 + i * 317);
                shape.graphics.drawCircle(0, 0, 7);
                shape.x = 35 + (i % 15) * 14;
                shape.y = 35 + int(i / 15) * 18;
                addChild(shape);
            }
            var label:TextField = new TextField();
            label.text = "Map / Inventory 0123456789";
            label.width = 230;
            label.x = 36; label.y = 142;
            label.filters = [new BlurFilter(2, 2)];
            addChild(label);
            var marker:Shape = new Shape();
            marker.graphics.beginFill(0x339966);
            marker.graphics.drawRect(0, 0, 24, 24);
            addChild(marker);
        }
    }
}
