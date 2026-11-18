package { import flash.display.*; import flash.events.*; import flash.filters.*; import flash.net.URLRequest;
 [SWF(width="1280",height="720",frameRate="60",backgroundColor="#ffffff")]
 public class Test extends Sprite {
  private var items:Array=[]; private var frame:int=0; private var sum:int=0; private var loader:Loader;
  public function Test() { for(var i:int=0;i<24;i++){var s:Sprite=box(i); var m:Sprite=box(i); m.x+=4; m.alpha=.5; s.cacheAsBitmap=true; m.cacheAsBitmap=true; addChild(s); addChild(m); s.mask=m;} addEventListener(Event.ENTER_FRAME,step); }
  private function step(event:Event):void { frame++;  }
  private function box(i:int):Sprite { var s:Sprite=new Sprite();s.graphics.beginFill(0x48a0dd,.7);s.graphics.drawRect(0,0,12,12);s.graphics.endFill();s.x=4+(i%80)*15;s.y=4+(int(i/80)%44)*15;return s; }
 }
}
class A { public var value:int=0; public function step():int {return value;} }
class B extends A { override public function step():int {return value+1;} }
class C extends A { override public function step():int {return value+2;} }
class D extends A { override public function step():int {return value+3;} }
class E extends A { override public function step():int {return value+4;} }
