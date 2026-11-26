package {
    import flash.display.Sprite;
    import flash.events.*;
    import flash.net.XMLSocket;

    public class Test extends Sprite {
        public function Test() {
            var socket:XMLSocket = new XMLSocket();
            var messages:int = 0;
            socket.addEventListener(DataEvent.DATA, function(e:DataEvent):void {
                trace(++messages, e.data);
                if (messages == 1) {
                    // Reconnect after the trailing partial message has been buffered.
                    addEventListener(Event.ENTER_FRAME, function reconnect(e:Event):void {
                        removeEventListener(Event.ENTER_FRAME, reconnect);
                        socket.connect("localhost", 8001);
                    });
                } else {
                    socket.close();
                }
            });
            socket.connect("localhost", 8001);
        }
    }
}
