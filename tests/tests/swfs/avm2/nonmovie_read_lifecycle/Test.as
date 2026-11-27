package {
    import flash.display.Sprite;
    import flash.events.*;
    import flash.media.Sound;
    import flash.net.*;

    public class Test extends Sprite {
        private var soundEvents:Array = [];
        private var streamEvents:Array = [];
        private var loaderEvents:Array = [];
        private var binaryEvents:Array = [];
        private var frames:int = 0;

        public function Test() {
            var sound:Sound = new Sound();
            sound.addEventListener(IOErrorEvent.IO_ERROR, function(e:Event):void { soundEvents.push(e.type); });
            sound.load(new URLRequest("missing.mp3"));
            sound.close();

            var playing:Sound = new Sound();
            playing.addEventListener(Event.OPEN, function(e:Event):void {
                soundEvents.push(e.type);
                playing.close();
            });
            playing.addEventListener(ProgressEvent.PROGRESS, function(e:Event):void { soundEvents.push(e.type); });
            playing.addEventListener(Event.ID3, function(e:Event):void { soundEvents.push(e.type); });
            playing.addEventListener(Event.COMPLETE, function(e:Event):void { soundEvents.push(e.type); });
            playing.load(new URLRequest("silence.mp3"));

            var nc:NetConnection = new NetConnection();
            nc.connect(null);
            var ns:NetStream = new NetStream(nc);
            ns.addEventListener(NetStatusEvent.NET_STATUS, function(e:NetStatusEvent):void {
                if (e.info.code == "NetStream.Buffer.Full") {
                    streamEvents.push(ns.bytesLoaded);
                    ns.close();
                }
            });
            ns.play("test.swf");
            ns.play("payload.txt");

            var loader:URLLoader = new URLLoader();
            loader.addEventListener(Event.OPEN, function(e:Event):void {
                loaderEvents.push(e.type);
                if (loaderEvents.length == 1) loader.load(new URLRequest("payload.txt"));
            });
            loader.addEventListener(ProgressEvent.PROGRESS, function(e:Event):void { loaderEvents.push(e.type); });
            loader.addEventListener(HTTPStatusEvent.HTTP_STATUS, function(e:Event):void { loaderEvents.push(e.type); });
            loader.addEventListener(Event.COMPLETE, function(e:Event):void { loaderEvents.push(loader.data); });
            loader.load(new URLRequest("test.swf"));

            var binary:URLStream = new URLStream();
            binary.addEventListener(ProgressEvent.PROGRESS, function(e:Event):void {
                binaryEvents.push(binary.readUnsignedByte());
                binary.close();
            });
            binary.addEventListener(Event.COMPLETE, function(e:Event):void { binaryEvents.push(e.type); });
            binary.load(new URLRequest("payload.txt"));

            addEventListener(Event.ENTER_FRAME, function(e:Event):void {
                if (++frames == 4) {
                    trace("sound", soundEvents.join(","));
                    trace("stream", streamEvents.join(","));
                    trace("loader", loaderEvents.join(","));
                    trace("binary", binaryEvents.join(","));
                }
            });
        }
    }
}
