package {
    import flash.display.DisplayObject;
    import flash.display.Loader;
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.net.URLRequest;

    [SWF(width="1280", height="720", frameRate="60", backgroundColor="#ffffff")]
    public class Test extends Sprite {
        private var loader:Loader = new Loader();
        private var retained:DisplayObject;
        private var frame:int;

        public function Test() {
            addChild(loader);
            loadChild();
            addEventListener(Event.ENTER_FRAME, step);
        }

        private function loadChild():void {
            loader.load(new URLRequest("child.swf"));
        }

        private function step(event:Event):void {
            frame++;
            if (frame == 12) {
                retained = loader.content;
                retained.x = 7.25;
                loader.unloadAndStop();
                trace("first child retained");
                loadChild();
            } else if (frame == 24) {
                if (loader.content == null || loader.content === retained) {
                    throw new Error("expected a separate loaded child");
                }
                loader.unloadAndStop();
                trace("second child unloaded");
            } else if (frame == 36) {
                trace("retained child accessible: " + (retained.x == 7.25));
                retained = null;
                trace("last external reference released");
            } else if (frame >= 72 && frame % 12 == 0) {
                loader.unloadAndStop();
                loadChild();
            }
        }
    }
}
