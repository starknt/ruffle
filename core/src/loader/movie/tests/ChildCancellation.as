package {
    import flash.display.Loader;
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.events.ProgressEvent;
    import flash.net.URLRequest;
    import flash.utils.ByteArray;

    public class ChildCancellation extends Sprite {
        private var replacing:Loader = new Loader();
        private var closed:Loader = new Loader();

        public function ChildCancellation() {
            // A complete 1x1 GIF. Local loadBytes reports progress synchronously.
            var gif:String = "47494638396101000100800000000000ffffff2c00000000010001000002024401003b";
            var bytes:ByteArray = new ByteArray();
            for (var i:int = 0; i < gif.length; i += 2) {
                bytes.writeByte(parseInt(gif.substr(i, 2), 16));
            }
            addChild(replacing);
            addChild(closed);
            replacing.contentLoaderInfo.addEventListener(ProgressEvent.PROGRESS, replaceFirst);
            replacing.contentLoaderInfo.addEventListener(Event.COMPLETE, function(e:Event):void {
                trace("complete " + replacing.content.width);
            });
            closed.contentLoaderInfo.addEventListener(Event.INIT, function(e:Event):void {
                trace("obsolete init");
            });
            closed.contentLoaderInfo.addEventListener(Event.COMPLETE, function(e:Event):void {
                trace("obsolete complete");
            });
            replacing.loadBytes(bytes);
            trace("replacement bytes " + replacing.contentLoaderInfo.bytesTotal);
            closed.loadBytes(bytes);
            closed.close(); // Both progress events ran, but frame delivery is still pending.
        }

        private function replaceFirst(e:ProgressEvent):void {
            replacing.contentLoaderInfo.removeEventListener(ProgressEvent.PROGRESS, replaceFirst);
            replacing.load(new URLRequest("test.png"));
        }
    }
}
