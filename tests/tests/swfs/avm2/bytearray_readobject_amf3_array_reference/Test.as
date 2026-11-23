package {
    import flash.display.Sprite;
    import flash.utils.ByteArray;

    public class Test extends Sprite {
        public function Test() {
            // The named property introduces an object. The dense entry then
            // refers to the same object in the AMF3 object table.
            var encoded:Array = [
                0x09, 0x03,                             // array with one dense entry
                0x09, 0x69, 0x74, 0x65, 0x6d,           // key "item"
                0x0a, 0x0b, 0x01,                       // anonymous dynamic object
                0x03, 0x6e, 0x04, 0x07, 0x01,           // n = 7, end of object
                0x03, 0x39, 0x03,                       // sparse index 9 = true
                0x01,                                   // end of associative entries
                0x0a, 0x02                              // reference to item
            ];
            var bytes:ByteArray = new ByteArray();
            for each (var byte:uint in encoded) {
                bytes.writeByte(byte);
            }
            bytes.position = 0;
            var result:Array = bytes.readObject() as Array;
            trace(result.length, result[9], result.item.n);
            trace(result[0] === result.item);
            result.item.n = 8;
            trace(result[0] && result[0].n);
        }
    }
}
