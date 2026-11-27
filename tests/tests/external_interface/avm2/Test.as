package {
    import flash.display.Sprite;
    import flash.external.ExternalInterface;

    public class Test extends Sprite {
        public function Test() {
            ExternalInterface.addCallback("callWith", forward);
            trace("outer received Traced!: " + (ExternalInterface.call("reentry") === "Traced!"));

            ExternalInterface.addCallback("callWith", fail);
            var result:* = ExternalInterface.call("reentry");
            trace("outer resumed with null: " + (result === null));
            trace("ping after throw: " + ExternalInterface.call("ping"));
        }

        private function forward(name:String, payload:String):* {
            return ExternalInterface.call(name, payload);
        }

        private function fail(name:String, payload:String):void {
            trace("callback throwing");
            throw "bridge-sentinel";
        }
    }
}
