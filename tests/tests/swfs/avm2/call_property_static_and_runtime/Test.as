package {
    import flash.display.Sprite;
    import flash.events.Event;

    public class Test extends Sprite {
        private var receiver:Object = {};
        private var calls:String = "";
        private var order:String = "";

        public function Test() {
            receiver.measuredStatic = function(value:int):int {
                calls += value;
                return value + 1;
            };
            addEventListener(Event.ENTER_FRAME, checkCalls);
        }

        private function key():String {
            order += "k";
            return "measuredStatic";
        }

        private function arg(value:int):int {
            order += "a";
            return value;
        }

        private function checkCalls(event:Event):void {
            removeEventListener(Event.ENTER_FRAME, checkCalls);
            var staticResult:int = receiver.measuredStatic(1);
            receiver.measuredStatic(2);
            var lateNameResult:int = receiver[key()](arg(3));
            receiver[key()](arg(4));
            var ns:Namespace = new Namespace("");
            var lateNamespaceResult:int = receiver.ns::measuredStatic(5);
            receiver.ns::measuredStatic(6);

            var caught:Boolean = false;
            try {
                receiver["missing"](arg(7));
            } catch (error:TypeError) {
                caught = true;
            }
            trace(staticResult, lateNameResult, lateNamespaceResult, calls, order, caught);
        }
    }
}
