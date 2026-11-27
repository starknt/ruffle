var ei = flash.external.ExternalInterface;

function forward(name, payload) {
    return ei.call(name, payload);
}

function fail() {
    trace("callback throwing");
    throw "bridge-sentinel";
}

ei.addCallback("callWith", this, forward);
trace("outer received Traced!: " + (ei.call("reentry") === "Traced!"));

ei.addCallback("callWith", this, fail);
var result = ei.call("reentry");
trace("outer resumed with null: " + (result === null));
trace("ping after throw: " + ei.call("ping"));
