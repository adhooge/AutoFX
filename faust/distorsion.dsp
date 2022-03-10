import("stdfaust.lib");

disto = gain,((_,drive):*:aa.tanh1):*;

// Parameters

drive = vslider("Drive[style:knob]", 1, 1, 10, 0.1);
gain = vslider("Gain[style:knob]", 1, 1, 5, 0.1);

process =  disto, disto ;