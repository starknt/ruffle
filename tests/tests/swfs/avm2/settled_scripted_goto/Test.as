package {
    import flash.display.MovieClip;
    import flash.events.Event;

    // Exercise repeated same-frame gotos after the initial script is consumed.
    // A nested goto queued by its replacement must retain the broadcast order.
    public class Test extends MovieClip {
        private var scripts:int = 0;
        private var constructed:int = 0;
        private var exited:int = 0;

        public function Test() {
            stop();
            addFrameScript(0, initial);
            addEventListener(Event.ENTER_FRAME, exercise);
        }

        private function initial():void {
            trace("initial " + (++scripts));
        }

        private function replacement():void {
            trace("replacement " + (++scripts));
            gotoAndStop(1);
            trace("after queued goto");
        }

        private function onConstructed(event:Event):void {
            trace("constructed " + (++constructed));
            if (constructed == 2) {
                addFrameScript(0, replacement);
            }
        }

        private function onExit(event:Event):void {
            trace("exit " + (++exited));
        }

        private function exercise(event:Event):void {
            // The first ENTER_FRAME can precede the initial frame script.
            if (scripts == 0) return;
            removeEventListener(Event.ENTER_FRAME, exercise);
            addEventListener(Event.FRAME_CONSTRUCTED, onConstructed);
            addEventListener(Event.EXIT_FRAME, onExit);
            for (var i:int = 0; i < 4; i++) {
                trace("goto " + i);
                gotoAndStop(1);
                trace("return " + i);
            }
            removeEventListener(Event.FRAME_CONSTRUCTED, onConstructed);
            removeEventListener(Event.EXIT_FRAME, onExit);
            trace("done scripts=" + scripts + " constructed=" + constructed + " exits=" + exited);
        }
    }
}
