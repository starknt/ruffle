package {
    import flash.display.MovieClip;
    public class Clip extends MovieClip {
        public var action:Function;
        private var label:String;
        public function Clip(label:String) {
            this.label = label;
            addFrameScript(0, script);
        }
        private function script():void {
            trace("script " + label);
            if (action != null) action();
        }
    }
}
