package {
    import flash.display.Bitmap;
    import flash.display.BitmapData;
    import flash.display.Sprite;
    import flash.filters.BitmapFilterType;
    import flash.filters.GradientBevelFilter;
    import flash.geom.Matrix;
    import flash.geom.Point;
    import flash.geom.Rectangle;

    [SWF(width="960", height="560", backgroundColor="#242424")]
    public class Test extends Sprite {
        public function Test() {
            var src:BitmapData = new BitmapData(192, 112, true, 0);
            src.fillRect(new Rectangle(16, 16, 160, 48), 0xff00ff00);
            src.fillRect(new Rectangle(16, 64, 160, 32), 0x8000ff00);
            var filter:GradientBevelFilter = new GradientBevelFilter(8, 0,
                [0xff0000, 0, 0x0000ff], [1, 0, 1], [0, 128, 255],
                1, 1, 1, 1, BitmapFilterType.FULL, false);

            // At these edges Flash Player places ratio 255 on the left at 0 degrees (blur 1, no smoothing).
            var normal:BitmapData = apply(src, filter);
            var bitmap:Bitmap = new Bitmap(normal);
            bitmap.x = 160;
            bitmap.y = 200;
            addChild(bitmap);

            // Trim the transparent border so the exterior bevel must expand the display cache.
            var tight:BitmapData = new BitmapData(160, 80, true, 0);
            tight.copyPixels(src, new Rectangle(16, 16, 160, 80), new Point());
            var live:Bitmap = new Bitmap(tight);
            live.x = live.y = 16;
            live.filters = [filter];
            var holder:Sprite = new Sprite();
            holder.addChild(live);
            holder.x = 608;
            holder.y = 200;
            addChild(holder);

            var display:BitmapData = new BitmapData(192, 112, true, 0);
            display.draw(holder);
            trace("display full exterior", blue(display.getPixel32(12, 40)), red(display.getPixel32(180, 40)));
            trace("display full interior", blue(display.getPixel32(20, 40)), red(display.getPixel32(172, 40)));
            // Move the child's geometry entirely outside the target, leaving only its right bevel visible.
            var fringe:BitmapData = new BitmapData(10, 64, true, 0xff243648);
            fringe.draw(holder, new Matrix(1, 0, 0, 1, -176, 0));
            trace("draw halo alone", red(fringe.getPixel32(3, 40)), fringe.getPixel32(9, 40) == 0xff243648);

            for each (var mode:String in [BitmapFilterType.INNER, BitmapFilterType.OUTER, BitmapFilterType.FULL]) {
                filter.type = mode;
                filter.knockout = false;
                normal = apply(src, filter);
                filter.knockout = true;
                var punched:BitmapData = apply(src, filter);
                var inside:Boolean = mode != BitmapFilterType.OUTER;
                var outside:Boolean = mode != BitmapFilterType.INNER;
                var left:uint = normal.getPixel32(20, 40);
                var right:uint = normal.getPixel32(172, 40);
                trace(mode, "normal interior", inside ? blue(left) && red(right) : left == 0xff00ff00 && right == 0xff00ff00);
                left = punched.getPixel32(20, 40);
                right = punched.getPixel32(172, 40);
                trace(mode, "knockout interior", inside ? blue(left) && red(right) : left == 0 && right == 0);
                for each (var result:BitmapData in [normal, punched]) {
                    left = result.getPixel32(12, 40);
                    right = result.getPixel32(180, 40);
                    trace(mode, result == normal ? "normal exterior" : "knockout exterior",
                        outside ? blue(left) && red(right) : left == 0 && right == 0);
                }
                trace(mode, "flat source alpha", normal.getPixel32(96, 40) == 0xff00ff00,
                    normal.getPixel32(96, 80) == 0x8000ff00, punched.getPixel32(96, 40) == 0,
                    punched.getPixel32(96, 80) == 0);
            }

            // Only the amount of source alpha changes between these two rows.
            var opaqueEdge:uint = punched.getPixel32(20, 40) >>> 24;
            var translucentEdge:uint = punched.getPixel32(20, 80) >>> 24;
            trace("translucent source weakens full bevel", translucentEdge > 0 && translucentEdge < opaqueEdge);
            filter.type = BitmapFilterType.INNER;
            var innerAlpha:uint = apply(src, filter).getPixel32(20, 80) >>> 24;
            filter.type = BitmapFilterType.OUTER;
            var outerAlpha:uint = apply(src, filter).getPixel32(20, 80) >>> 24;
            trace("translucent source masks knockout", innerAlpha > 0 && innerAlpha < translucentEdge,
                outerAlpha > 0 && outerAlpha < translucentEdge);

            filter.type = BitmapFilterType.FULL;
            filter.angle = 180;
            punched = apply(src, filter);
            trace("180 degrees swaps sides", red(punched.getPixel32(20, 40)), blue(punched.getPixel32(172, 40)));
            filter.strength = 0.5;
            var weak:uint = apply(src, filter).getPixel32(20, 40) >>> 24;
            trace("weaker bevel reduces edge alpha", weak > 0 && weak < opaqueEdge);

            filter.angle = 0;
            filter.strength = 1;
            filter.distance = 2;
            var dot:BitmapData = new BitmapData(1, 1, true, 0xff00a000);
            var dst:BitmapData = new BitmapData(6, 1, true, 0xff00a000);
            dst.applyFilter(dot, dot.rect, new Point(-1, 0), filter);
            trace("offscreen source casts visible right edge", red(dst.getPixel32(1, 0)));
            trace("outside destination bounds preserved", dst.getPixel32(5, 0) == 0xff00a000);

            var strip:BitmapData = new BitmapData(24, 8, true, 0);
            strip.fillRect(new Rectangle(7, 2, 4, 4), 0xff00a000);
            strip.setPixel32(0, 4, 0xff00a000);
            strip.applyFilter(strip, new Rectangle(4, 2, 12, 4), new Point(6, 2), filter);
            // Moving the crop two pixels right puts its original green run at x=9..12.
            trace("in-place crop moves both edges", blue(strip.getPixel32(8, 4)), red(strip.getPixel32(13, 4)));
            trace("in-place crop preserves distant pixel", strip.getPixel32(0, 4) == 0xff00a000);

            // Equal masks use the middle stop, including at zero strength.
            filter.distance = 1;
            filter.strength = 0;
            filter.colors = [0xff0000, 0x00ff00, 0x0000ff];
            filter.alphas = [1, 1, 1];
            var tile:BitmapData = new BitmapData(9, 1, true, 0);
            tile.setPixel32(4, 0, 0xff000000);
            normal = apply(tile, filter);
            trace("zero strength middle stop", normal.getPixel32(3, 0).toString(16), normal.getPixel32(4, 0).toString(16));
            filter.strength = 1;
            filter.colors = [0, 0xffffff];
            filter.alphas = [1, 1];
            filter.ratios = [0, 255];
            for each (var alpha:uint in [1, 127, 128, 192, 254, 255]) {
                tile.setPixel32(4, 0, alpha << 24);
                normal = apply(tile, filter);
                trace("signed gray", alpha, normal.getPixel32(3, 0) & 255, normal.getPixel32(4, 0) & 255, normal.getPixel32(5, 0) & 255);
            }
            filter.colors = [0xff0000, 0x00ff00, 0x0000ff];
            filter.alphas = [1, 1, 1];
            filter.ratios = [0, 128, 255];
            filter.distance = 0.5;
            normal = apply(tile, filter);
            trace("half pixel ramp colors", normal.getPixel32(3, 0).toString(16), normal.getPixel32(5, 0).toString(16));
            filter = new GradientBevelFilter(1, 0, [], [], [], 1, 1, 1, 1, BitmapFilterType.FULL);
            trace("empty ramp keeps source", apply(tile, filter).getPixel32(4, 0).toString(16));
            filter.knockout = true;
            trace("empty ramp knockout", apply(tile, filter).getPixel32(4, 0).toString(16));
        }

        private function apply(src:BitmapData, filter:GradientBevelFilter):BitmapData {
            var dst:BitmapData = new BitmapData(src.width, src.height, true, 0);
            dst.applyFilter(src, src.rect, new Point(), filter);
            return dst;
        }

        private function blue(pixel:uint):Boolean {
            return (pixel >>> 24) >= 250 && (pixel & 255) >= 250 && ((pixel >>> 8) & 0xffff) == 0;
        }

        private function red(pixel:uint):Boolean {
            return (pixel >>> 24) >= 250 && ((pixel >>> 16) & 255) >= 250 && (pixel & 0xffff) == 0;
        }
    }
}
