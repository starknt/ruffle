package {
    import flash.display.Bitmap;
    import flash.display.BitmapData;
    import flash.display.Sprite;
    import flash.filters.ConvolutionFilter;
    import flash.geom.Point;
    import flash.geom.Rectangle;

    public class Test extends Sprite {
        public function Test() {
            var src:BitmapData = new BitmapData(5, 5, true, 0xff204060);
            src.setPixel32(1, 3, 0xff6080a0);
            src.setPixel32(1, 4, 0xff80a0c0);
            src.setPixel32(2, 2, 0x40000000);
            src.setPixel32(2, 3, 0x80000000);
            src.setPixel32(2, 4, 0x80000000);

            // Matrix entries 4 and 7 average (x, y) and (x, y + 1).
            var filter:ConvolutionFilter = new ConvolutionFilter(3, 3,
                [0, 0, 0,
                 0, 1, 0,
                 0, 1, 0], 2);
            var dst:BitmapData = apply(src, filter);
            trace("interior", dst.getPixel32(2, 1).toString(16), dst.getPixel32(2, 2).toString(16));
            trace("clamp", dst.getPixel32(2, 3).toString(16));
            trace("preserved alpha", dst.getPixel32(3, 1) >>> 24);
            trace("outside destination", dst.getPixel32(1, 1), dst.getPixel32(2, 4));

            filter.preserveAlpha = false;
            filter.clamp = false;
            dst = apply(src, filter);
            trace("filtered alpha", dst.getPixel32(3, 1) >>> 24);
            trace("default fill", dst.getPixel32(3, 3).toString(16));

            filter.color = 0x204080;
            filter.alpha = 1;
            dst = apply(src, filter);
            trace("color fill", dst.getPixel32(2, 3).toString(16));

            filter.bias = 7;
            var sprite:Sprite = new Sprite();
            sprite.filters = [filter];
            var attached:ConvolutionFilter = ConvolutionFilter(sprite.filters[0]);
            var copy:ConvolutionFilter = ConvolutionFilter(attached.clone());
            trace("display clone", copy.matrixX, copy.matrixY, copy.matrix, copy.divisor,
                copy.bias, copy.preserveAlpha, copy.clamp, copy.color.toString(16), copy.alpha);
            copy.matrix[4] = 77;
            trace("independent clone", attached.matrix[4]);

            filter.matrixX = 31;
            filter.matrixY = -1;
            trace("bounded dimensions", filter.matrixX, filter.matrixY);

            var strip:BitmapData = new BitmapData(3, 1, true, 0);
            strip.setPixel32(0, 0, 0x8000ff00);
            strip.setPixel32(1, 0, 0xff080808);
            strip.setPixel32(2, 0, 0xff0a0a0a);
            var opaque:BitmapData = new BitmapData(3, 1, false, 0xffffff);
            opaque.applyFilter(strip, strip.rect, new Point(), new ConvolutionFilter(1, 1, [1], 0));
            trace("opaque destination", opaque.getPixel32(0, 0).toString(16));
            opaque.applyFilter(strip, strip.rect, new Point(), new ConvolutionFilter(1, 1, [1], 3, 1));
            trace("truncate after bias", opaque.getPixel(1, 0).toString(16), opaque.getPixel(2, 0).toString(16));
            addChild(new Bitmap(opaque));
            var doubled:Bitmap = new Bitmap(opaque);
            doubled.y = 1;
            doubled.filters = [new ConvolutionFilter(1, 1, [2])];
            addChild(doubled);

            var clipped:BitmapData = new BitmapData(4, 1, true, 0xff112233);
            var identity:ConvolutionFilter = new ConvolutionFilter(1, 1, [1]);
            clipped.applyFilter(strip, new Rectangle(-1, 0, 3, 1), new Point(), identity);
            clipped.applyFilter(strip, new Rectangle(99, 0, 1, 1), new Point(), identity);
            trace("clipped region", clipped.getPixel32(0, 0).toString(16), clipped.getPixel32(1, 0).toString(16),
                clipped.getPixel32(2, 0).toString(16), clipped.getPixel32(3, 0).toString(16));
            var halfStep:BitmapData = new BitmapData(1, 1, true, 0xccb2b2b2);
            var straight:BitmapData = new BitmapData(1, 1, false, 0);
            straight.applyFilter(halfStep, halfStep.rect, new Point(), identity);
            trace("straight channel rounding", straight.getPixel32(0, 0).toString(16));
        }

        private function apply(src:BitmapData, filter:ConvolutionFilter):BitmapData {
            var dst:BitmapData = new BitmapData(6, 5, true, 0);
            dst.applyFilter(src, new Rectangle(1, 2, 3, 3), new Point(2, 1), filter);
            return dst;
        }
    }
}
