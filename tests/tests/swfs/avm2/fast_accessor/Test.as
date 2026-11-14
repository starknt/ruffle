package {
    import flash.display.MovieClip;

    public class Test extends MovieClip {
        public var probe:Object;
        public function Test() {
            var base:Base = new Base();
            probe = base;
            trace("initial", base.value);
            base.field = 19;
            trace("updated", base.value);
            var alias:Base = base;
            trace("alias", alias.value);
            var child:Base = new Child();
            trace("override", child.value);
            child.field = 23;
            trace("override updated", child.value);
            trace("effects", base.sideEffect, base.sideEffect, base.calls);
            try { trace(base.failure); } catch (e:Error) { trace("caught", e.message); }
            var payload:Payload = new Payload();
            base.untyped = payload;
            trace("coerce", base.text, base.text, payload.calls);
            var bound:Function = base.read;
            trace("bound", bound(), bound());
            try { bound(99); } catch (e:ArgumentError) { trace("argument", e.errorID); }
            trace("after error", base.value);
            stop();
        }
    }
}

class Base {
    public var field:int = 7;
    public var calls:int = 0;
    public var untyped:*;
    public function get value():int { return field; }
    public function get freshValue():int { return field; }
    public function read():int { return field; }
    public function get sideEffect():int { return ++calls; }
    public function get failure():int { throw new Error("getter error"); }
    public function get text():String { return untyped; }
}

class Child extends Base {
    override public function get value():int { return field + 100; }
}

class Payload {
    public var calls:int = 0;
    public function toString():String { calls++; return "converted"; }
}
