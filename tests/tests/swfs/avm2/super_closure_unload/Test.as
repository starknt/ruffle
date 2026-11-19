package {
    import flash.display.Loader;
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.net.URLRequest;

    public class Test extends Sprite {
        private var loader:Loader = new Loader();
        public function Test() {
            loader.contentLoaderInfo.addEventListener(Event.COMPLETE, loaded);
            loader.load(new URLRequest("child.swf"));
        }
        private function loaded(event:Event):void {
            var child:Object = loader.content;
            var keep:Function = child.step;
            trace(keep === child.step, keep(2));
            child = null;
            loader.unloadAndStop();
            trace(keep(7));
        }
    }
}
