package {
    import flash.display.Sprite;
    import flash.events.EventDispatcher;
    import flash.text.engine.*;

    public class Test extends Sprite {
        public function Test() {
            var mirror:EventDispatcher = new EventDispatcher();
            var format:ElementFormat = new ElementFormat();
            var elements:Array = [
                new TextElement("abc", format, mirror, TextRotation.ROTATE_90),
                new GroupElement(null, format, mirror, TextRotation.ROTATE_180),
                new GraphicElement(null, 15, 15, format, mirror, TextRotation.ROTATE_270)
            ];
            for each (var element:ContentElement in elements) {
                trace(element.elementFormat === format, element.eventMirror === mirror, element.textRotation);
            }
            var defaults:TextElement = new TextElement();
            trace(defaults.eventMirror, defaults.textRotation);
        }
    }
}
