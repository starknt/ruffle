package  {
	
	import flash.display.MovieClip;
	import flash.display.Stage3D;
	import flash.events.Event;
	import flash.display3D.Context3D;
	import flash.display3D.Context3DProfile;
	
	
	public class Test extends MovieClip {
		
		var PROFILES: Array = [Context3DProfile.BASELINE_CONSTRAINED, Context3DProfile.BASELINE, Context3DProfile.BASELINE_EXTENDED, Context3DProfile.STANDARD_CONSTRAINED, Context3DProfile.STANDARD, Context3DProfile.STANDARD_EXTENDED];
		var stage3D: Stage3D;
		var previous: Context3D;
		var creations: int;
		
		public function Test() {
			this.addEventListener(Event.ENTER_FRAME, this.onEnterFrame);
			this.stage.stage3Ds[0].addEventListener(Event.CONTEXT3D_CREATE, onCreate);
			/*try {
				this.stage.stage3Ds[0].requestContext3D("auto", "bad");
			} catch (e) {
				trace("Caught e: " + e);
			}*/

		}
	
		private function onCreate(e:Event):void {
			creations++;
		}

		private function onEnterFrame(e) {
			if (!this.stage3D) {
				var profile = PROFILES.pop();
				trace("Requesting profile: " + profile);
				this.stage3D = this.stage.stage3Ds[0];
				this.stage3D.requestContext3D("auto", profile);
			} else if (this.stage3D.context3D) {
				var current:Context3D = stage3D.context3D;
				trace("Got profile: " + current.profile);
				if (PROFILES.length == 0) {
					previous.dispose(false);
					trace("Stale dispose(false) kept successor: " + (stage3D.context3D === current));
					previous.dispose(true);
					trace("Stale dispose(true) kept successor: " + (stage3D.context3D === current));
					previous = current;
					current.dispose();
					trace("Immediately after dispose(): " + stage3D.context3D);
					current.dispose(false);
					removeEventListener(Event.ENTER_FRAME, onEnterFrame);
					addEventListener(Event.ENTER_FRAME, checkRecreation);
					return;
				}
				previous = current;
				current.dispose(false);
				trace("After dispose: " + this.stage3D.context3D);
				this.stage3D = null;
			}
		}

		private function checkRecreation(e:Event):void {
			var current:Context3D = stage3D.context3D;
			if (current == null) {
				return;
			}
			trace("Recreated a new context: " + (current !== previous));
			trace("Recreated profile: " + current.profile);
			trace("Creation events: " + creations);
			stage3D.removeEventListener(Event.CONTEXT3D_CREATE, onCreate);
			current.dispose(true);
			removeEventListener(Event.ENTER_FRAME, checkRecreation);
			addEventListener(Event.ENTER_FRAME, checkWithoutListener);
		}

		private function checkWithoutListener(e:Event):void {
			trace("After dispose(true) without listener: " + stage3D.context3D);
			removeEventListener(Event.ENTER_FRAME, checkWithoutListener);
		}
	}
	
}
