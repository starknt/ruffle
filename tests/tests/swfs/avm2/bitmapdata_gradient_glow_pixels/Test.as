package {
    import flash.display.Bitmap;
    import flash.display.BitmapData;
    import flash.display.Sprite;
    import flash.filters.BitmapFilterType;
    import flash.filters.GradientGlowFilter;
    import flash.geom.Point;
    import flash.geom.Rectangle;

    [SWF(width="960", height="560", backgroundColor="#242424")]
    public class Test extends Sprite {
        public function Test() {
            var src:BitmapData = new BitmapData(80, 56, true, 0);
            src.fillRect(new Rectangle(12, 8, 20, 40), 0xffcccccc);
            src.fillRect(new Rectangle(32, 30, 24, 18), 0x80cccccc);

            var filter:GradientGlowFilter = new GradientGlowFilter(12, 0,
                [0xff0000, 0xff0000, 0x0000ff], [0, 1, 1], [0, 128, 255],
                4, 4, 1, 1, BitmapFilterType.OUTER, true);
            addChild(new Bitmap(src));
            var filtered:Bitmap = new Bitmap(src);
            filtered.x = 80;
            filtered.filters = [filter];
            addChild(filtered);

            var dst:BitmapData = apply(src, filter);
            // Twelve pixels to the right, the shifted foot samples red and the stem blue.
            var foot:uint = dst.getPixel32(62, 38);
            var stem:uint = dst.getPixel32(38, 18);
            trace("outer half-alpha foot is red", isGlow(foot) && ((foot >> 16) & 255) > (foot & 255));
            trace("outer opaque stem is blue", isGlow(stem) && (stem & 255) > ((stem >> 16) & 255));
            trace("outer knockout clears overlap", dst.getPixel32(28, 18) == 0);

            // At x=24 the shifted, blurred left edge crosses the original opaque stem.
            filter.type = BitmapFilterType.INNER;
            dst = apply(src, filter);
            trace("inner colors the stem", isGlow(dst.getPixel32(24, 18)));
            trace("inner masks the exterior", dst.getPixel32(38, 18) == 0);

            filter.type = BitmapFilterType.FULL;
            dst = apply(src, filter);
            trace("full colors both sides", isGlow(dst.getPixel32(24, 18)) && isGlow(dst.getPixel32(38, 18)));

            var dot:BitmapData = new BitmapData(1, 1, true, 0xff0000ff);
            var offset:GradientGlowFilter = new GradientGlowFilter(3, 0,
                [0, 0xff0000], [0, 1], [0, 255], 1, 1, 1, 1, BitmapFilterType.OUTER, true);
            dst = new BitmapData(6, 1, true, 0xff00ff00);
            dst.applyFilter(dot, dot.rect, new Point(-1, 0), offset);
            trace("offscreen source casts visible glow", dst.getPixel32(2, 0) == 0xffff0000);
            trace("outside glow bounds untouched", dst.getPixel32(3, 0) == 0xff00ff00);
            var strip:BitmapData = new BitmapData(8, 1, true, 0);
            strip.setPixel32(1, 0, 0xff0000ff);
            offset.distance = 2;
            strip.applyFilter(strip, new Rectangle(3, 0, 1, 1), new Point(4, 0), offset);
            trace("in-place subrectangle samples its neighbor", strip.getPixel32(4, 0) == 0xffff0000);
            trace("in-place source neighbor preserved", strip.getPixel32(1, 0) == 0xff0000ff);
            offset.distance = -2;
            dst.fillRect(dst.rect, 0xff00ff00);
            dst.applyFilter(dot, dot.rect, new Point(4, 0), offset);
            trace("negative distance casts left", dst.getPixel32(2, 0) == 0xffff0000);
            offset.strength = 0;
            dst.applyFilter(dot, dot.rect, new Point(4, 0), offset);
            trace("zero strength copies source", dst.getPixel32(4, 0) == 0xff0000ff);

            var edge:BitmapData = new BitmapData(2, 1, true, 0);
            edge.setPixel32(0, 0, 0x800000ff);
            var solid:GradientGlowFilter = new GradientGlowFilter(0, 0, [0xff0000], [1], [0],
                1, 1, 1, 1, BitmapFilterType.INNER, false);
            for each (var mode:String in [BitmapFilterType.INNER, BitmapFilterType.OUTER, BitmapFilterType.FULL]) {
                solid.type = mode;
                solid.knockout = false;
                var normal:uint = apply(edge, solid).getPixel32(0, 0);
                solid.knockout = true;
                var punched:uint = apply(edge, solid).getPixel32(0, 0);
                trace(mode, "normal alpha", normal >>> 24, "knockout alpha", punched >>> 24);
            }
            trace("full opaque first stop colors transparent pixels", apply(edge, solid).getPixel32(1, 0) == 0xffff0000);
            solid.colors = [0, 0x00ff00, 0xff0000];
            solid.alphas = [0, 1, 1];
            solid.ratios = [0, 128, 255];
            solid.strength = 2;
            trace("strength scales lookup before color selection", apply(edge, solid).getPixel32(0, 0) == 0xffff0000);

            var opaque:BitmapData = new BitmapData(2, 1, false, 0xffffff);
            solid.type = BitmapFilterType.INNER;
            opaque.applyFilter(edge, edge.rect, new Point(), solid);
            trace("opaque destination keeps red at full opacity", opaque.getPixel32(0, 0) == 0xffff0000);
            solid.strength = 0;
            opaque.applyFilter(edge, edge.rect, new Point(), solid);
            trace("opaque identity keeps source blue", opaque.getPixel32(0, 0) == 0xff0000ff);

            var ramp:GradientGlowFilter = new GradientGlowFilter(0, 0, [0, 0xffffff], [1, 1], [0, 255],
                5, 1, 1, 1, BitmapFilterType.FULL, true);
            var impulse:BitmapData = new BitmapData(15, 1, true, 0);
            impulse.setPixel32(7, 0, 0xffffffff);
            var weak:uint = apply(impulse, ramp).getPixel(7, 0) & 255;
            ramp.strength = 2;
            var strong:uint = apply(impulse, ramp).getPixel(7, 0) & 255;
            trace("strength scales blurred alpha", weak > 0 && weak < 127 && Math.abs(strong - 2 * weak) <= 1);
            ramp.blurX = 1;
            ramp.distance = 0.5;
            trace("fractional offset mixes alpha before strength", apply(impulse, ramp).getPixel(8, 0) == 0xffffff);

            // The displayed red pixel lies five pixels outside the one-pixel bitmap.
            var marker:Bitmap = new Bitmap(dot);
            marker.x = 96;
            marker.y = 51;
            offset.strength = 1;
            offset.distance = 5;
            marker.filters = [offset];
            addChild(marker);
            var tiny:GradientGlowFilter = new GradientGlowFilter(0, 0, [0, 0xffffff], [1, 1], [0, 255],
                3, 1, 1, 1, BitmapFilterType.INNER, true);
            var tinyStage:Bitmap = new Bitmap(dot);
            tinyStage.x = 105;
            tinyStage.y = 51;
            tinyStage.filters = [tiny];
            addChild(tinyStage);
            var tinyBitmap:Bitmap = new Bitmap(apply(dot, tiny));
            tinyBitmap.x = 108;
            tinyBitmap.y = 51;
            addChild(tinyBitmap);

            offset.distance = 1.0 / 32.0;
            offset.colors = [0, 0xff0000];
            offset.ratios = [0, 7];
            dst.fillRect(dst.rect, 0);
            dst.applyFilter(dot, dot.rect, new Point(), offset);
            trace("sub-twip offset reaches adjacent pixel", dst.getPixel32(1, 0) == 0xffff0000);
            offset.distance = 32000;
            for each (var blur:Number in [1, 3]) {
                offset.blurX = blur;
                dst.fillRect(dst.rect, 0);
                dst.applyFilter(dot, dot.rect, new Point(-32000, 0), offset);
                trace("distant single pixel", blur, dst.getPixel32(0, 0) == 0xffff0000);
            }
        }

        private function apply(src:BitmapData, filter:GradientGlowFilter):BitmapData {
            var dst:BitmapData = new BitmapData(src.width, src.height, true, 0);
            dst.applyFilter(src, src.rect, new Point(), filter);
            return dst;
        }

        private function isGlow(pixel:uint):Boolean {
            // The red/blue ramp must be visible against both transparency and the gray source.
            var green:uint = (pixel >> 8) & 255;
            return (pixel >>> 24) > 16 && Math.max((pixel >> 16) & 255, pixel & 255) > green + 32;
        }
    }
}
