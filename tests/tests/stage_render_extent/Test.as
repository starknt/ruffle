package {
    import flash.display.*;
    import flash.display3D.*;
    import flash.events.Event;
    import flash.filters.BlurFilter;
    import flash.geom.Point;
    import flash.geom.Rectangle;
    import flash.utils.ByteArray;
    import flash.utils.Endian;

    [SWF(width="105", height="73", backgroundColor="#000000")]
    public class Test extends Sprite {
        public function Test() {
            stage.scaleMode = StageScaleMode.NO_SCALE;
            stage.align = StageAlign.TOP_LEFT;
            trace("viewport", stage.stageWidth, stage.stageHeight);

            var clip:Sprite = new Sprite();
            clip.x = 28;
            clip.y = 10;
            clip.scrollRect = new Rectangle(0, 0, 41, 48);
            addChild(clip);
            var blurred:Sprite = rectangle(8, 8, 48, 32, 0xff0000);
            blurred.cacheAsBitmap = true;
            blurred.filters = [new BlurFilter(8, 8, 1)];
            blurred.blendMode = BlendMode.DIFFERENCE;
            clip.addChild(blurred);
            var stencil:Sprite = rectangle(20, 24, 66, 34, 0xffffff);
            addChild(stencil);
            clip.mask = stencil;

            // The drawn rectangle extends beyond both reduced stage attachments.
            var drawn:BitmapData = new BitmapData(99, 69, true, 0);
            drawn.draw(rectangle(83, 57, 16, 12, 0x00ff00));
            checkRect("draw", drawn, new Rectangle(83, 57, 16, 12), 0xff00ff00, 0);
            var copied:BitmapData = new BitmapData(37, 27, true, 0);
            copied.copyPixels(drawn, new Rectangle(79, 53, 20, 16), new Point(11, 7));
            checkRect("copy", copied, new Rectangle(15, 11, 16, 12), 0xff00ff00, 0);
            trace("pixel and hit", copied.getPixel32(30, 22).toString(16),
                copied.hitTest(new Point(), 1, new Point(30, 22)),
                copied.hitTest(new Point(), 1, new Point(31, 22)));

            var s3d:Stage3D = stage.stage3Ds[0];
            s3d.x = 7;
            s3d.y = 5;
            s3d.addEventListener(Event.CONTEXT3D_CREATE, onContext);
            s3d.requestContext3D();
        }

        private function rectangle(x:int, y:int, w:int, h:int, color:uint):Sprite {
            var sprite:Sprite = new Sprite();
            sprite.graphics.beginFill(color);
            sprite.graphics.drawRect(x, y, w, h);
            sprite.graphics.endFill();
            return sprite;
        }

        private function checkRect(label:String, bitmap:BitmapData, rect:Rectangle,
                                   inside:uint, outside:uint):void {
            var pixels:Vector.<uint> = bitmap.getVector(bitmap.rect);
            var errors:int = 0;
            for (var y:int = 0; y < bitmap.height; y++) {
                for (var x:int = 0; x < bitmap.width; x++) {
                    if (pixels[y * bitmap.width + x] != (rect.contains(x, y) ? inside : outside)) {
                        errors++;
                    }
                }
            }
            trace(label, bitmap.width, bitmap.height, "mismatches", errors);
        }

        // AGAL 1: mov op, va0 (vertex) or mov oc, fc0 (fragment).
        private function shader(fragment:Boolean):ByteArray {
            var b:ByteArray = new ByteArray();
            b.endian = Endian.LITTLE_ENDIAN;
            b.writeByte(0xa0);
            b.writeUnsignedInt(1);
            b.writeByte(0xa1);
            b.writeByte(fragment ? 1 : 0);
            b.writeUnsignedInt(0);          // mov
            b.writeUnsignedInt(0x030f0000); // output register, xyzw
            b.writeUnsignedInt(0xe4000000); // identity swizzle
            b.writeUnsignedInt(fragment ? 1 : 0);
            b.writeUnsignedInt(0);
            b.writeUnsignedInt(0);
            b.position = 0;
            return b;
        }

        private function onContext(event:Event):void {
            var ctx:Context3D = Stage3D(event.target).context3D;
            ctx.configureBackBuffer(83, 61, 0, false);
            var program:Program3D = ctx.createProgram();
            program.upload(shader(false), shader(true));
            ctx.setProgram(program);
            ctx.setProgramConstantsFromVector("fragment", 0, Vector.<Number>([1, 1, 0, 1]));
            var vertices:VertexBuffer3D = ctx.createVertexBuffer(4, 2);
            vertices.uploadFromVector(Vector.<Number>([-1,-1, 1,-1, -1,1, 1,1]), 0, 4);
            ctx.setVertexBufferAt(0, vertices, 0, Context3DVertexBufferFormat.FLOAT_2);
            var indices:IndexBuffer3D = ctx.createIndexBuffer(6);
            indices.uploadFromVector(Vector.<uint>([0,1,2, 2,1,3]), 0, 6);
            var scissor:Rectangle = new Rectangle(57, 37, 18, 17);
            ctx.clear(0, 0, 1, 1);
            ctx.setScissorRectangle(scissor);
            ctx.drawTriangles(indices);
            var read:BitmapData = new BitmapData(89, 67, true, 0xff808080);
            ctx.drawToBitmapData(read);
            ctx.present();
            // Check the scissor in logical coordinates, then check the entire source
            // footprint and the untouched right and bottom margins of the destination.
            var back:BitmapData = new BitmapData(83, 61);
            back.copyPixels(read, back.rect, new Point());
            checkRect("Stage3D scissor", back, scissor, 0xffffff00, 0xff0000ff);
            read.fillRect(scissor, 0xff0000ff);
            checkRect("drawToBitmapData", read, back.rect, 0xff0000ff, 0xff808080);
        }
    }
}
