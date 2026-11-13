# Reentrant construction invalidation

Stock-Ruffle characterization; expected output is not a new Flash capture.
Root frame 1 places a two-frame MovieClip named left. Root frame 2 places an
empty DefineShape linked to Trigger. The left timeline has an empty frame 1
and places another empty shape on frame 2. Both clips initially stop.

After the first passes settle the tree, root advances. Trigger's constructor
adds a Sprite to left (already visited by the current construction traversal),
starts left and stops root. On the following frame, left's newly placed timeline
shape must be constructed. Without the mutation-revision check, the outer pass
incorrectly marks root clean while left is dirty; the next invalidation stops at
left and the trace reports an unconstructed shape. This was reproduced by
removing just that check, then passed again with the check restored.

ASC inputs are Test.as and Trigger.as with Test as document class. The SWF adds
DefineSprite character 1 (left, two frames), empty DefineShape characters 2 and 3,
a SymbolClass mapping 3 to Trigger, and the placements described above. The
sources and the hand-authored timeline together define this fixture.
