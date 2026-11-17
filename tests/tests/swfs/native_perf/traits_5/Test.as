package { import flash.display.*; import flash.events.*; import flash.filters.*; import flash.net.URLRequest;
 [SWF(width="1280",height="720",frameRate="60",backgroundColor="#ffffff")]
 public class Test extends Sprite {
  private var items:Array=[]; private var frame:int=0; private var sum:int=0; private var loader:Loader;
  public function Test() { items=[new A(),new B(),new C(),new D(),new E()].slice(0,5); addEventListener(Event.ENTER_FRAME,step); }
  private function step(event:Event):void { frame++; for(var i:int=0;i<12000;i++){var x:Object=items[i%items.length]; x.value=(x.value+1)&255;sum+=x.step();} }
  private function box(i:int):Sprite { var s:Sprite=new Sprite();s.graphics.beginFill(0x48a0dd,.7);s.graphics.drawRect(0,0,12,12);s.graphics.endFill();s.x=4+(i%80)*15;s.y=4+(int(i/80)%44)*15;return s; }
 }
}
class A { public var value:int=0; public function step():int {return value;} }
class B extends A { override public function step():int {return value+1;} }
class C extends A { override public function step():int {return value+2;} }
class D extends A { override public function step():int {return value+3;} }
class E extends A { override public function step():int {return value+4;} }
