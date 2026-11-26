package {
    import flash.display.MovieClip;
    import flash.events.Event;
    import flash.net.NetConnection;
    import flash.net.ObjectEncoding;
    import flash.net.Responder;
    import flash.utils.setTimeout;

    public class Test extends MovieClip {
        private const url:String = "http://localhost/batch";
        private var connection:NetConnection = new NetConnection();
        private var frame:int = 0;

        public function Test() {
            connection.objectEncoding = ObjectEncoding.AMF0;
            addEventListener(Event.ENTER_FRAME, onFrame);
        }

        private function call(label:String):void {
            connection.call("test", new Responder(
                function(value:*):void { trace(label, "result", value); },
                function(value:*):void { trace(label, "status", value); }
            ));
        }

        private function onFrame(event:Event):void {
            // The test runner flushes calls, runs timers, then delivers the local reply.
            switch (++frame) {
                case 1:
                    connection.connect(url);
                    call("stale after close");
                    setTimeout(function():void {
                        connection.close();
                        trace("closed");
                    }, 1);
                    break;
                case 2:
                    connection.connect(url);
                    call("stale after reconnect");
                    setTimeout(function():void {
                        connection.connect(url);
                        trace("reconnected");
                        call("replacement");
                        call("replacement");
                    }, 1);
                    break;
                case 4:
                    connection.call("test", new Responder(function(value:*):void {
                        trace("closing in responder", value);
                        connection.close();
                    }));
                    call("stale within batch");
                    break;
            }
        }
    }
}
