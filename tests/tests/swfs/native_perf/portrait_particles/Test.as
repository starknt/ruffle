package {
    import flash.display.BlendMode;
    import flash.display.Shape;
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.filters.BlurFilter;
    import flash.filters.ColorMatrixFilter;
    import flash.filters.GlowFilter;

    [SWF(width="256", height="144", frameRate="60", backgroundColor="#253248")]
    public class Test extends Sprite {
        private var particles:Vector.<Shape> = new Vector.<Shape>();
        private var tick:int = 0;

        public function Test() {
            for (var i:int = 0; i < 48; ++i) {
                var p:Shape = new Shape();
                p.graphics.beginFill(0xffe090);
                p.graphics.drawEllipse(-3, -2, 6, 4);
                p.x = 13 + (i % 12) * 20;
                p.y = 16 + int(i / 12) * 28;
                if (i % 3 == 0) {
                    p.filters = [new BlurFilter(5, 5), new ColorMatrixFilter([
                        0.82, -0.25, 0.43, 0, 0, 0.09, 1.03, -0.12, 0, 0,
                        -0.33, 0.40, 0.93, 0, 0, 0, 0, 0, 1, 0
                    ])];
                } else {
                    p.filters = [new GlowFilter(0xffcc00, 1, 2, 2, 1.02),
                                 new GlowFilter(0xffff33, 1, 4, 4, 0.83),
                                 new GlowFilter(0xffff00, 1, 5, 5, 1.30)];
                }
                addChild(p);
                particles.push(p);
            }

            // Transparent stencil geometry still clips the stripe. Alpha
            // blending on a layer must also keep its normal compositing rules.
            var stripe:Sprite = new Sprite();
            stripe.graphics.beginFill(0x40d0e0);
            stripe.graphics.drawRect(0, 120, 256, 12);
            addChild(stripe);
            var stencil:Sprite = new Sprite();
            stencil.graphics.beginFill(0xffffff);
            stencil.graphics.drawRect(16, 120, 100, 12);
            stencil.alpha = 0;
            stencil.cacheAsBitmap = true;
            addChild(stencil);
            stripe.mask = stencil;
            var layer:Sprite = new Sprite();
            layer.blendMode = BlendMode.LAYER;
            layer.graphics.beginFill(0xff8040);
            layer.graphics.drawRect(140, 120, 96, 12);
            var cutout:Sprite = new Sprite();
            cutout.graphics.beginFill(0xffffff);
            cutout.graphics.drawRect(164, 120, 48, 12);
            cutout.cacheAsBitmap = true;
            cutout.alpha = 0;
            cutout.blendMode = BlendMode.ALPHA;
            layer.addChild(cutout);
            addChild(layer);
            addEventListener(Event.ENTER_FRAME, advance);
        }

        private function advance(event:Event):void {
            ++tick;
            for (var i:int = 0; i < particles.length; ++i) {
                var p:Shape = particles[i];
                var phase:int = (tick + i) % 12;
                p.rotation = phase * 7;
                p.alpha = tick % 24 < 6 ? 0 : (phase + 1) / 12;
                if (tick % 24 == 3 && i == 0) {
                    p.graphics.clear();
                    p.graphics.beginFill((tick % 48 == 3) ? 0xe03060 : 0x50e060);
                    p.graphics.drawRect(-3, -2, 7, 5);
                }
            }
        }
    }
}
