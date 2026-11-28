package {
    import flash.display.*;
    import flash.events.*;
    import flash.net.URLRequest;

    public class Test extends Sprite {
        private var loader:Loader;
        private var cycle:int;
        private var age:int;
        private var visibleChild:Boolean;
        private var unloaded:Boolean;

        public function Test() {
            graphics.beginFill(0x202020);
            graphics.drawRect(0, 0, 2048, 2048);
            graphics.endFill();
            addEventListener(Event.ENTER_FRAME, step);
        }
        private function step(event:Event):void {
            if (!loader) {
                if (++age < 3) return;
                age = 0;
                cycle++;
                loader = new Loader();
                addChild(loader);
                loader.contentLoaderInfo.addEventListener(Event.COMPLETE, loaded);
                loader.contentLoaderInfo.addEventListener(Event.UNLOAD, didUnload);
                loader.load(new URLRequest("child.swf"));
            } else if (visibleChild) {
                if (++age < 4) return;
                visibleChild = false;
                loader.unloadAndStop();
            } else if (unloaded && loader.content == null) {
                removeChild(loader);
                loader = null;
                unloaded = false;
                age = 0;
                trace("memory-unloaded", cycle);
            }
        }
        private function loaded(event:Event):void {
            if (loader.content == null || !loader.content.stage) throw new Error("child missing from stage");
            visibleChild = true;
            age = 0;
            trace("memory-loaded", cycle);
        }
        private function didUnload(event:Event):void { unloaded = true; }
    }
}
