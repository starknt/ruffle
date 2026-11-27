package {
    import flash.display.Sprite;
    import flash.errors.IOError;
    import flash.events.Event;
    import flash.events.IOErrorEvent;
    import flash.events.ProgressEvent;
    import flash.net.Socket;
    import flash.utils.ByteArray;

    public class Test extends Sprite {
        private var socket:Socket = new Socket();

        public function Test() {
            socket.addEventListener(Event.CONNECT, connected);
            socket.addEventListener(ProgressEvent.SOCKET_DATA, acknowledged);
            socket.addEventListener(Event.CLOSE, function(e:Event):void { trace("closed"); });
            socket.addEventListener(IOErrorEvent.IO_ERROR, function(e:Event):void { trace(e); });
            socket.connect("localhost", 8001);
        }

        private function connected(e:Event):void {
            var body:ByteArray = new ByteArray();
            body.length = 8 * 1024 * 1024 + 17;
            socket.writeBytes(body);
            socket.flush();
            socket.writeUTFBytes("retry once");
            try {
                socket.flush();
                trace("unexpected second flush");
            } catch (error:IOError) {
                trace("backpressure", error.errorID, socket.connected);
            }
        }

        private function acknowledged(e:ProgressEvent):void {
            trace("writer released", socket.readUnsignedByte());
            socket.flush();
        }
    }
}
