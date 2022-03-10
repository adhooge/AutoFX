import("stdfaust.lib");


lfo = depth * os.osc(rate);

// GUI
rate = vslider("Rate (Hz)[style:knob]", 20, 1, 100, 0.01);
mix = vslider("Dry/Wet[style:knob]", 0.5, 0, 1, 0.01);
delay = vslider("Centre delay (ms)[style:knob]", 7, 1, 100, 0.1);
depth = vslider("Depth[style:knob]", 0.2, 0.01, 0.99, 0.01);
feedback = vslider("Feedback[style:knob]", 0.2, -1, 1, 0.01);

delay_line = @( ((delay*lfo) + delay)*ma.SR/1000) : *(feedback) ;

chorus = _ <: _, +~delay_line : *(1-mix), *(mix) :+ ;

process =  chorus, chorus ;