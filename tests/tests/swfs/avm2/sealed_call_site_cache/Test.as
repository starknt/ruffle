package {
 import flash.display.Sprite;
 public class Test extends Sprite {
  public function Test() {
   for(var i:int=0;i<4;i++) { probe(new A()); probe(new B()); }
   var a:Object=new A();
   trace("read", read(a),read(a));
   var dynamicObject:Object={prop:31};
   trace("dynamic",read(dynamicObject)); dynamicObject.prop=44; trace("dynamic",read(dynamicObject));
   trace("read",read(a));
  }
  private function read(x:Object):int { return x.prop; }
  private function probe(x:Object):void {
   x.slot="7"; x.prop=8;
   var f:Function=x.method;
   trace(x.prop,x.method(),f===x.method,f(),x.slot,x.calls);
   x.ping(); trace("void",x.slot);
   try { x.c=22; } catch(e:Error) { trace("constant",e.errorID); }
   try { trace(x.failure); } catch(e:Error) { trace("getter",e.message); }
   try { x.method(3); } catch(e:Error) { trace("arity",e.errorID); }
   trace("after",x.method());
  }
 }
}
class A {
 public var slot:int=1;
 public var calls:int=0;
 public const c:int=9;
 public function get prop():int { calls++; return slot; }
 public function set prop(v:int):void { slot=v+1; }
 public function method():int { return slot; }
 public function ping():void { slot++; }
 public function get failure():int { throw new Error("boom"); }
}
class B extends A {
 override public function get prop():int { calls++; return slot+100; }
 override public function method():int { return slot+100; }
}
