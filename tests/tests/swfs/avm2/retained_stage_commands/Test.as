package {
 import flash.display.*;
 import flash.events.Event;
 import flash.geom.ColorTransform;
 import flash.geom.Rectangle;
 [SWF(width="256", height="64", frameRate="30", backgroundColor="#ffffff")]
 public class Test extends Sprite {
  private var group:Sprite = new Sprite();
  private var items:Array = [];
  private var frame:int = 0;
  public function Test() {
   addChild(group);
   for (var i:int=0; i<8; i++) {
    var s:Sprite = new Sprite();
    s.graphics.beginFill(i == 0 ? 0xff0000 : 0x00ff00);
    s.graphics.drawRect(0,0,16,16); s.graphics.endFill();
    s.x=4+i*28; s.y=4; group.addChild(s); items.push(s);
   }
   addEventListener(Event.ENTER_FRAME, step);
   step(null);
  }
  private function step(e:Event):void {
   frame++;
   if (frame==3) {
    items[0].x=24;
    items[1].graphics.clear(); items[1].graphics.beginFill(0x0000ff);
    items[1].graphics.drawRect(0,0,16,16); items[1].graphics.endFill();
   }
   if (frame==4) group.setChildIndex(items[0],7);
   if (frame==5) {
    var bitmap:BitmapData=new BitmapData(256,64,false,0xffffff);
    bitmap.draw(group); bitmap.dispose();
    items[0].transform.colorTransform=new ColorTransform(0,0,0,1,255,255,0,0);
   }
   if (frame==6) { removeChild(group); group.x=8; addChild(group); }
   if (frame==7) group.transform.colorTransform=new ColorTransform(1,0,1,1,0,0,0,0);
   if (frame==8) items[2].visible=false;
   if (frame==9) group.scrollRect=new Rectangle(0,0,120,64);
   if (frame==10) group.scrollRect=null;
   trace(frame);
  }
 }
}
