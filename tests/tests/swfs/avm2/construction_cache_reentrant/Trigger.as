package {
    import flash.display.Shape;
    import flash.display.Sprite;
    public class Trigger extends Shape {
        public function Trigger() {
            // left was already visited in this construction pass. Its next
            // timeline frame must still construct after this insertion.
            Test.instance.left.addChild(new Sprite());
            Test.instance.left.play();
            Test.instance.stop();
            trace("mutate-left");
        }
    }
}
