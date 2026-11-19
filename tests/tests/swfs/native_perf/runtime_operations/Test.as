package {
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.text.engine.*;

    public class Test extends Sprite {
        private var group:Sprite = new Sprite();
        private var counter:Derived = new Derived();
        private var text:String = new Array(512).join("ab") + "needle";
        private var sum:int;
        public function Test() {
            var first:Sprite = new Sprite();
            var second:Sprite = new Sprite();
            first.name = "other";
            second.name = "match";
            group.addChild(first);
            group.addChild(second);
            trace(group.getChildByName("match") === second);
            first.name = "match";
            trace(group.getChildByName("match") === first);
            group.swapChildren(first, second);
            trace(group.getChildByName("match") === second, group.getChildByName("Match") === null);

            for each (var value:String in [null, "", "abcd"]) {
                var element:TextElement = new TextElement(value);
                var elements:GroupElement = new GroupElement(Vector.<ContentElement>([element]));
                try {
                    var split:TextElement = elements.splitTextElement(0, 0);
                    trace(elements.elementCount, element.text.length, elements.getElementAt(1) === split, split.text);
                } catch (e:Error) {
                    trace(e.errorID);
                }
            }
            for (var i:int = 0; i < 200; ++i) group.addChild(new Sprite());
            var last:Sprite = new Sprite();
            last.name = "last";
            group.addChild(last);
            addEventListener(Event.ENTER_FRAME, step);
        }
        private function step(e:Event):void {
            for (var i:int = 0; i < 100; i++) {
                var whole:String = text.substr(0);
                sum += whole.indexOf("needle") + whole.lastIndexOf("missing");
                sum += counter.step(i);
                if (group.getChildByName("last") != null) sum++;
            }
        }
    }
}
class Parent {
    public function add(x:int):int { return x + 1; }
}
class Derived extends Parent {
    public function step(x:int):int { return super.add(x); }
}
