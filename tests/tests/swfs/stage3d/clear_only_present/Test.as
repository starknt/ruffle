package {
    import flash.display.Sprite;
    import flash.display.Stage3D;
    import flash.display3D.Context3D;
    import flash.events.Event;

    [SWF(width="64", height="64", backgroundColor="#000000")]
    public class Test extends Sprite {
        public function Test() {
            var s3d:Stage3D = stage.stage3Ds[0];
            s3d.addEventListener(Event.CONTEXT3D_CREATE, onContext);
            s3d.requestContext3D();
        }

        private function onContext(e:Event):void {
            var context:Context3D = Stage3D(e.target).context3D;
            context.configureBackBuffer(64, 64, 0, false);
            context.clear(1, 0, 0, 1);
            context.present();
        }
    }
}
