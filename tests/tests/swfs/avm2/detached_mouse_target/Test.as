package {
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.events.MouseEvent;
    import flash.utils.Dictionary;

    public class Test extends Sprite {
        private var panels:Dictionary = new Dictionary(true);
        private var panel:Sprite;
        private var frame:int = 0;

        public function Test() {
            mount("first", 0);
            addEventListener(Event.ENTER_FRAME, advance);
        }

        private function mount(label:String, x:int):void {
            panel = new Sprite();
            var button:Sprite = new Sprite();
            button.name = label;
            button.tabEnabled = false;
            button.graphics.beginFill(0xcc6633);
            button.graphics.drawRect(0, 0, 50, 50);
            button.graphics.endFill();
            button.addEventListener(MouseEvent.MOUSE_OVER, mouse);
            button.addEventListener(MouseEvent.MOUSE_DOWN, mouse);
            if (label == "second") {
                button.addEventListener(MouseEvent.MOUSE_OUT, mouse);
                button.addEventListener(MouseEvent.RELEASE_OUTSIDE, mouse);
            }
            panel.x = x;
            panel.addChild(button);
            addChild(panel);
            panels[panel] = true;
        }

        private function unmount():void {
            removeChild(panel);
            panel = null;
        }

        private function mouse(event:MouseEvent):void {
            trace(event.type, event.currentTarget.name);
        }

        private function panelCount():int {
            var count:int = 0;
            for (var key:* in panels) ++count;
            return count;
        }

        private function advance(event:Event):void {
            switch (++frame) {
                case 3:
                    trace("on stage", panelCount());
                    unmount();
                    break;
                case 8: trace("idle after removal", panelCount()); break;
                case 15: trace("after motion", panelCount()); break;
                case 16: mount("second", 95); break;
                case 20: unmount(); break;
                case 25: trace("button still pressed", panelCount()); break;
                case 32: trace("after release", panelCount()); break;
            }
        }
    }
}
