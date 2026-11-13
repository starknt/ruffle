package {
    import flash.display.MovieClip;
    import flash.display.Shape;
    import flash.display.SimpleButton;
    import flash.events.Event;

    public class Test extends MovieClip {
        private var tick:int = 0;
        public function Test() {
            var root:Test = this;
            var leaf:Shape = new Shape();
            leaf.addEventListener(Event.ENTER_FRAME, function(e:Event):void {
                trace("leaf enter");
            });
            addChild(leaf);
            var a:Clip = new Clip("a");
            var b:Clip = new Clip("b");
            addChild(a);
            addChild(b);
            a.action = function():void {
                trace("mutate");
                removeChild(b);
                addChild(new Clip("c"));
                swapChildrenAt(0, numChildren - 1);
                a.action = null;
            };
            var button:SimpleButton = new SimpleButton();
            button.upState = new Clip("button-up");
            button.hitTestState = new Clip("button-hit");
            addChild(button);
            addEventListener(Event.ENTER_FRAME, function(e:Event):void {
                trace("root enter " + (++tick));
            });
            addEventListener(Event.EXIT_FRAME, function(e:Event):void {
                trace("root exit");
            });
        }
    }
}
