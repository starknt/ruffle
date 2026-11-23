package {
    import flash.display.Loader;
    import flash.display.Sprite;
    import flash.events.HTTPStatusEvent;
    import flash.events.IOErrorEvent;
    import flash.net.URLRequest;
    import flash.utils.ByteArray;

    public class Test extends Sprite {
        private var request:Loader = new Loader();

        public function Test() {
            request.contentLoaderInfo.addEventListener(IOErrorEvent.IO_ERROR, function(e:IOErrorEvent):void {
                trace("obsolete error");
            });
            if (loaderInfo.url.indexOf("https:") == 0) {
                var bytes:ByteArray = new ByteArray();
                bytes.writeUTFBytes("missing content");
                request.loadBytes(bytes);
                request.loadBytes(bytes);
                request.close();
                trace("closed byte loads");
            } else {
                request.contentLoaderInfo.addEventListener(HTTPStatusEvent.HTTP_STATUS, function(e:HTTPStatusEvent):void {
                    trace("status");
                    request.close();
                    trace("closed");
                });
                request.load(new URLRequest("missing.swf"));
            }
        }
    }
}
