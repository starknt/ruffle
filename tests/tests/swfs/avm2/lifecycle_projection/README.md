# Lifecycle projection regression

This is a stock-Ruffle characterization test, not a new Flash reference capture.
The expected trace was obtained with the unfiltered lifecycle path. It asserts
that a removed sibling still runs from the current snapshot, newly created clips
run their frame script, both SimpleButton state timelines are visited, and Shape
enterFrame broadcast listeners keep firing. Existing Flash-derived frame, goto,
button and loader fixtures remain the compatibility oracle.

Compile Test.as with ASC, importing playerglobal_import.abc, using
`-AS3 -in Clip.as -swf Test,320,200,30 Test.as`. The SWF additionally contains a
DefineSprite with character ID 1 and two empty ShowFrame tags, followed by an End,
and a SymbolClass mapping ID 1 to Clip, before the root ShowFrame. This gives Clip
a real timeline: plain AS-created MovieClips have no frames and would make the
frame-script assertions vacuous.
