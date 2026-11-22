package flash.utils {
    [Ruffle(InstanceAllocator)]
    public dynamic class Dictionary {
        prototype.toJSON = function(r:String):* {
            return "Dictionary";
        };
        prototype.setPropertyIsEnumerable("toJSON", false);

        public function Dictionary(weakKeys:Boolean = false) {
            if (weakKeys) {
                useWeakKeys();
            }
        }

        private native function useWeakKeys():void;
    }
}
