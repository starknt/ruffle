package {
    import flash.display.MovieClip;
    import flash.events.Event;
    public class Test extends MovieClip {
        public static var instance:Test;
        public var left:MovieClip;
        private var tick:int = 0;
        public function Test() {
            instance = this;
            stop();
            left.stop();
            addEventListener(Event.ENTER_FRAME, function(e:Event):void {
                if (++tick == 2) play();
            });
            addEventListener(Event.EXIT_FRAME, function(e:Event):void {
                var states:Array = [];
                for (var i:int = 0; i < left.numChildren; i++) {
                    states.push(left.getChildAt(i) == null ? "unconstructed" : "constructed");
                }
                trace("exit " + left.currentFrame + " [" + states.join(",") + "]");
            });
        }
    }
}
