package {
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.utils.ByteArray;
    import flash.utils.Dictionary;

    public class Test extends Sprite {
        private var weak:Dictionary = new Dictionary(true);
        private var strong:Dictionary = new Dictionary();
        private var values:Dictionary = new Dictionary(true);
        private var held:Object = {};
        private var tick:int = 0;

        public function Test() {
            trace("AMF3 weak flags", flag(weak), flag(strong));
            var buffer:ByteArray = new ByteArray();
            buffer.writeObject(weak);
            buffer.position = 0;
            weak = buffer.readObject() as Dictionary;
            var dead:Object = {};
            var value:Object = {};
            weak[dead] = value;
            values[value] = "value";
            weak[held] = "live";
            weak["string"] = "string";
            strong[{}] = "strong";
            var selfKey:Object = {};
            weak[selfKey] = selfKey;
            var removed:Object = {};
            weak[removed] = "deleted";
            delete weak[removed];
            trace("deleted", removed in weak, weak[removed]);
            addEventListener(Event.ENTER_FRAME, check);
        }

        private function flag(dictionary:Dictionary):int {
            var buffer:ByteArray = new ByteArray();
            buffer.writeObject(dictionary);
            // The dictionary marker and single-byte entry count precede the flag.
            return buffer[2];
        }

        private function check(event:Event):void {
            if (++tick != 10) return;
            // The dictionary has been idle since the constructor. Its dead key
            // and the value owned only by that entry have both had time to go.
            var count:int = 0;
            for each (var observed:* in values) ++count;
            trace("values retained for dead keys", count);
            var listed:Array = [];
            for (var key:* in weak) {
                listed.push(weak[key] === key && typeof key == "object"
                    ? "self" : weak[key]);
            }
            trace("live entries", listed.sort());
            trace("external key", held in weak, weak[held]);
            for each (var kept:* in strong) trace("strong entry", kept);
        }
    }
}
