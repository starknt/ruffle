package {
    import flash.display.*;
    import flash.events.*;
    import flash.net.URLRequest;

    [SWF(width="1280", height="720", frameRate="60", backgroundColor="#ffffff")]
    public class Test extends Sprite {
        private var loader:Loader = new Loader();
        private var snapshotGroup:DisplayObjectContainer;
        private var frames:int;
        private var completed:int;
        public function Test() {
            addChild(loader);
            loader.contentLoaderInfo.addEventListener(Event.COMPLETE, loaded);
            loader.load(new URLRequest("child.swf"));
            addEventListener(Event.ENTER_FRAME, step);
        }
        private function loaded(event:Event):void {
            // A Loader blocks retained-command snapshots below it. Keep the
            // child's static group behind the loader as a sibling until unload.
            var child:DisplayObjectContainer = DisplayObjectContainer(loader.content);
            snapshotGroup = DisplayObjectContainer(child.getChildByName("snapshotGroup"));
            addChildAt(snapshotGroup, 0);
            completed++;
            if (completed <= 3) trace("loaded", completed, snapshotGroup.numChildren);
        }
        private function step(event:Event):void {
            frames++;
            if (frames % 120 == 0) {
                if (snapshotGroup) {
                    removeChild(snapshotGroup);
                    snapshotGroup = null;
                }
                loader.unloadAndStop();
                loader.load(new URLRequest("child.swf"));
            }
        }
    }
}
