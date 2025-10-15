//
// MIDI control for ACE
//
"ACE: loading ctl ... ".post;


~setVal = {|keyname,val|
	p[keyname] = val; // update parameter vector
	if( x.isPlaying, {x.set(keyname, p[keyname])}); // update Synth
	{ // update GUI
		if( g.expertGUI.sliders.includesKey(keyname), {
			g.expertGUI.sliders[keyname].value = p[keyname];
		});
		if( g.simpleGUI.sliders.includesKey(keyname), {
			g.simpleGUI.sliders[keyname].value = p[keyname];
		});
		if( g.simpleGUI.includesKey(\xypad), {
			if((keyname==\rho)||(keyname==\beta), {
				g.simpleGUI.xypad.y = ~rhobeta2scp.(p.rho, p.beta);
			});
			if(keyname==\t60atCutoff, {
				g.simpleGUI.xypad.x = p.t60atCutoff/2;
			});
		});
	}.defer; // push through the AppClock to be able to update the GUI
	(keyname++": "++val).postln;
};


~startmidi = {
	MIDIClient.init(1,0,false); // initialize midi client
	MIDIIn.connectAll(false); // connect to all possible midi inputs
	~midiFunc = MIDIFunc.cc({ |val,cc,ch,dev|
		// [val,cc,ch,dev].postln; // debug print all incoming
		switch(cc,
			// Settings for KORG nanoKONTROL 1
			13, { // fader 9
				~setVal.(\vol, val.linlin(0, 127, -70, 30));
			},
			12, { // fader 8
				~setVal.(\wet, val.linlin(0, 127, 0, 1));
			},
			9, { // fader 7
				~setVal.(\t60atCutoff, val.linlin(0, 127, 0, 2));
			},
			8, { // fader 6
				~setVal.(\rho, ~scp2rho.(val.linlin(0, 127, 0, 1)));
				~setVal.(\beta, ~scp2beta.(val.linlin(0, 127, 0, 1)));
			},
			6, { // fader 5
			},
			5, { // fader 4
			},
			4, { // fader 3
			},
			3, { // fader 2
			},
			2, { // fader 1

			},
			14, { // rotary 1
			},
			15, { // rotary 2
			},
			16, { // rotary 3
			},
			17, { // rotary 4
			},
			18, { // rotary 5
			},
			19, { // rotary 6
			},
			20, { // rotary 7
			},
			21, { // rotary 8
			},
			22, { // rotary 9

			}
		);
	});
};


~stopmidi = {
	~midiFunc.free; // cleanup
	MIDIIn.disconnectAll; // disconnect from all midi inputs
	MIDIClient.disposeClient; // clean up midi client
};



"done.".postln;