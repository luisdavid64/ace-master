//
// GUI for ACE
//
"ACE: loading GUI ... ".post;

// -------------------- Expert GUI --------------------
~expertGUI = {

	// name min max warp step default units
	// 1    2   3   4    5    6       7
	var sliderList = [
		[\fHPFtc, 50, 20000, \lin, 1, "Hz", "TCE cutoff frequency"],
		[\tauAtc, 0, 99, \lin, 0.0, "ms", "TCE attack time constant"],
		[\tauDtc, 0, 99, \lin, 0.0, "ms", "TCE decay time constant"],
		[\nu, -100, 0, \lin, 1, "dB", "TCE threshold"],

		[\fHSF, 0, 20000, \lin, 1, "Hz", "High-Shelf frequency"],
		[\sHSF, 0.000001, 1, \lin, 0.0, "", "High-Shelf slope"],
		[\dbHSF, 0, 40, \lin, 1, "dB", "High-Shelf gain"],

		[\dbNoise, -96, 0, \lin, 1, "dB", "noise level"],

		[\rho, 0, 30, \lin, 0.0, "", "LI strength ρ"],
		[\tauLI, 0, 70, \lin, 0.0, "ms", "LI smoothing time constant"],

		[\beta, 0, 15, \lin, 0.0, "", "EX strength β"],
		[\mu, -60, 0, \lin, 1, "dB", "EX threshold"],
		[\tauEX, 0, 70, \lin, 0.0, "ms", "EX smoothing time constant"],

		[\tauAdp, 0, 70, \lin, 0.0, "ms", "DP attack time constant"],
		[\t60atCutoff, 0, 2, \lin, 0.0, "s", "DP RT60 at cutoff frequency"],
		[\dpCutoff, 0, 20000, \lin, 1, "Hz", "DP cutoff frequency"],

		[\dBregDelta, -96, 0, \lin, 1, "dB", "delta for regularization"],
		[\tauSP, 0, 10, \lin, 0.0, "ms", "Smoothing time constant"],
		[\wet, 0, 1, \lin, 0.0, "", "dry - wet"],
		[\dimWeight, 0, 1, \lin, 0.0, "", "TCE - SCE"],
		[\vol, -60, 10, \lin, 0.0, "dB", "Volume"]
	];

	var makeSlider = { |item|
		EZSlider(elements.w, 750@32, "  "++item[6]++"  ", ControlSpec(item[1],item[2],item[3],item[4],p[item[0]],item[5]), labelWidth: 250, unitWidth: 30, initVal: p[item[0]], numberWidth: 60, layout: \horz)
		.setColors(Color.grey, Color.white)
		.font_(Font("Helvetica", 16))
		.action_({ |ez|
			p[item[0]] = ez.value;
			x.isPlaying.if{x.set(item[0], p[item[0]])};
			if( g.simpleGUI.sliders.includesKey(item[0]), {
				g.simpleGUI.sliders[item[0]].value = p[item[0]];
			});
			if( g.simpleGUI.includesKey(\xypad), {
				if((item[0]==\rho)||(item[0]==\beta), {
					g.simpleGUI.xypad.y = ~rhobeta2scp.(p.rho, p.beta);
				});
				if(item[0]==\t60atCutoff, {
					g.simpleGUI.xypad.x = p.t60atCutoff/2;
				});
			});
		});
	};



	var elements = ();

	elements.w = Window("ACE-RT | Expert GUI", Rect(400,400,758,760))
	.visible_(true)
	.alwaysOnTop_(true);

	elements.v = elements.w.view
	.decorator_(FlowLayout(elements.w.view.bounds))
	.decorator.gap_(2@2);

	elements.state = Button(elements.w, Rect(0,0,50,30))
	.states_([["OFF", Color.white, Color.red], ["ON", Color.gray, Color.green]])
	.action_({ |obj|
		var val = obj.value;
		if(
			val==1,
			{if(x.isPlaying, {}, {x = Synth.head(s, \ace, p.getPairs).register;});},
			{x.set(\gate, 0);}
		);
		if(g.simpleGUI.notNil, {g.simpleGUI.state.value = val;});
		if(g.interGUI.notNil, {g.interGUI.state.value = val;});
	});

	elements.testRec = Button(elements.w, Rect(0,0,110,30))
	.states_([["REC stopped", Color.green, Color.gray], ["REC started", Color.white, Color.red]])
	.action_({|obj|
		var val=obj.value;
		if(
			val==1,
			{if(~testRecSynth.isPlaying, {}, {
				q.testRecPath = (~path.pathOnly++"wav/rec_"++Date.getDate.format("%Y-%m-%d_%H-%M-%S")++".wav").standardizePath;
				q.testRecBuf.write(q.testRecPath, "WAV", "int24", 0, 0, true); // create file, leave open
				~testRecSynth = Synth.tail(s, \ace_testrec, [\buf, q.testRecBuf.bufnum, \inOffset, ~inOffset, \outOffset, ~outOffset]).register; // record
				("ACE: recording started ("++q.testRecPath++")").postln;
			};);},
			{
				~testRecSynth.free;
				q.testRecBuf.close;
				("ACE: recording stopped ("++q.testRecPath++")").postln;
			}
		);
	});

	elements.sliders = ().putPairs(sliderList.collect{|item,i| [item[0], makeSlider.(item)]}.flatten);

	elements;
};



// -------------------- Simple GUI --------------------
~simpleGUI = {

	// name min max warp step default units
	// 1    2   3   4    5    6       7
	var sliderList = [
		[\wet, 0, 1, \lin, 0.0, "", "dry - wet"],
		[\dimWeight, 0, 1, \lin, 0.0, "", "TCE - SCE"],
		[\vol, -70, 30, \lin, 0.0, "dB", "Volume"]
	];

	var makeSlider = { |item|
		EZSlider(elements.w, 400@25, "  "++item[6]++"  ", ControlSpec(item[1],item[2],item[3],item[4],p[item[0]],item[5]), labelWidth: 100, unitWidth: 30, initVal: p[item[0]], numberWidth: 60, layout: \horz)
		.setColors(Color.grey, Color.white)
		.font_(Font("Helvetica", 16))
		.action_({ |ez|
			p[item[0]] = ez.value;
			if( x.isPlaying, {x.set(item[0], p[item[0]])});
			if( g.expertGUI.sliders.includesKey(item[0]), {
				g.expertGUI.sliders[item[0]].value = p[item[0]];
			});
		});
	};

	var elements = ();

	elements.w =  Window("ACE-RT | Simple GUI", Rect(400,400,408,560))
	.visible_(true)
	.alwaysOnTop_(true);

	elements.v = elements.w.view
	.decorator_(FlowLayout(elements.w.view.bounds))
	.decorator.gap_(2@2);

	elements.state = Button(elements.w, Rect(0,0,50,30))
	.states_([["OFF", Color.white, Color.red], ["ON", Color.gray, Color.green]])
	.action_({ |obj|
		var val = obj.value;
		if(
			val==1,
			{if(x.isPlaying, {}, {x = Synth.head(s, \ace, p.getPairs).register;};);},
			{x.set(\gate, 0);}
		);
		if(g.expertGUI.notNil, {g.expertGUI.state.value = val;});
		if(g.interGUI.notNil, {g.interGUI.state.value = val;});
	});

	elements.sliders = ().putPairs(sliderList.collect{|item,i| [item[0], makeSlider.(item)]}.flatten);

	elements.xypad = Slider2D(elements.w, Rect(0, 0, 400, 400))
	.y_(~rhobeta2scp.(p.rho, p.beta)) // initial position of y
	.x_(p.t60atCutoff/2) // initial location of x
	.action_({ |sl|
		var rho = ~scp2rho.(sl.y);
		var beta = ~scp2beta.(sl.y);
		var t60atCutoff = 2*sl.x;
		x.isPlaying.if{x.set(\beta, beta, \rho, rho, \t60atCutoff, t60atCutoff)};
		p.rho = rho;
		p.beta = beta;
		p.t60atCutoff = t60atCutoff;
		if( g.expertGUI.notNil, {
			if( g.expertGUI.sliders.includesKey(\beta), {
				g.expertGUI.sliders.beta.value = beta;
			});
			if( g.expertGUI.sliders.includesKey(\rho), {
				g.expertGUI.sliders.rho.value = rho;
			});
			if( g.expertGUI.sliders.includesKey(\t60atCutoff), {
				g.expertGUI.sliders.t60atCutoff.value = t60atCutoff;
			});
		});
	});

	elements.menu = EZPopUpMenu(elements.w, 400@32, "  Presets:  ", items: p.presets.flop[0],
		globalAction: { |item,i|
			elements.xypad.setXYActive(
				p.presets[item.value][1],
				~rhobeta2scp.(p.presets[item.value][2][0], p.presets[item.value][2][1])
		)},
		initVal: 6
	)
	.setColors(Color.grey, Color.white)
	.font_(Font("Helvetica", 16));

	elements;
};


// -------------------- INTER-STIMULUS ACE GUI --------------------


//a = q.validMagsdb.plot;
//a.

~interGUI = {

	var updatePlotMags = {|plotMags|
		q.validIndices = [q.binFreqs.detectIndex{|item,i| item>=50}, q.binFreqs.size - 1 - q.binFreqs.reverse.detectIndex{|item,i| item<=20000}];
		q.validMagsdb = q.sampleMags.flop[q.validIndices[0]..q.validIndices[1]].flop.ampdb;
		q.validMagsMaxdb = q.validMagsdb.flat.maxItem;
		q.validMagsMindb = q.validMagsdb.flat.minItem;
		q.validBinFreqs = q.binFreqs[q.validIndices[0]..q.validIndices[1]];
		plotMags
		.setValue(q.validMagsdb-q.validMagsMaxdb)
		.superpose_(true)
		.plotColor_([Color.blue, Color.red])
		.domainSpecs_([q.validBinFreqs.first, q.validBinFreqs.last, \exp, 0].asSpec)
		//.domain_(q.validBinFreqs)
		.setProperties(
			\gridOnX, false
		)
		.specs_([q.validMagsMindb-q.validMagsMaxdb, 0].asSpec)
		.refresh;
	};

	var updatePlotDiff = {|plotDiff|
		plotDiff
		.setValue(q.magDiffdb[q.validIndices[0]..q.validIndices[1]]-q.magDiffdbMax)
		.domainSpecs_([q.validBinFreqs.first, q.validBinFreqs.last, \exponential, 0].asSpec)
		//.domain_(q.validBinFreqs)
		.setProperties(
			\gridOnX, false
		)
		.specs_([q.magDiffdbMax.neg, 0].asSpec)
		.refresh;
	};

	var updatePlotWeights = {|plotWeights|
		plotWeights
		.setValue(q.bandWeights.ampdb)
		.domainSpecs_([q.validBinFreqs.first, q.validBinFreqs.last, \exponential, 0].asSpec)
		.domain_(q.fc)
		.setProperties(
			\gridOnX, false
		)
		.specs_([-100, 0].asSpec)
		.plotMode_(\points)
		.refresh;
	};

	var elements = ();

	elements.w = Window("Inter-stimulus ACE-RT", Rect(400,400,560,650))
	.visible_(true)
	.alwaysOnTop_(true);

	elements.v = elements.w.view
	.decorator_(FlowLayout(elements.w.view.bounds))
	.decorator.gap_(2@2);

	elements.state = Button(elements.w, Rect(0,0,50,30))
	.states_([["OFF", Color.white, Color.red], ["ON", Color.gray, Color.green]])
	.action_({ |obj|
		var val = obj.value;
		if(
			val==1,
			{if(x.isPlaying, {}, {x = Synth.head(s, \ace, p.getPairs).register;};);},
			{x.set(\gate, 0);}
		);
		if(g.expertGUI.notNil, {g.expertGUI.state.value = val;});
		if(g.simpleGUI.notNil, {g.simpleGUI.state.value = val;});
	});


	elements.recA = Button(elements.w, Rect(0,0,90,30))
	.states_([["A READY", Color.green, Color.gray], ["A REC", Color.white, Color.red]])
	.action_({|obj|
		var val=obj.value;
		var path = (~path.pathOnly++"wav/bufA.wav").standardizePath;
		if(
			val==1,
			{if(y[0].isPlaying, {}, {
				q.sampleBufs[0].write(path, "wav", "float", 0, 0, true); // create file, leave open
				y[0] = Synth.tail(s, \ace_rec, [\buf, q.sampleBufs[0].bufnum, \inOffset, ~inOffset]).register; // record
			};);},
			{
				y[0].free;
				q.sampleBufs[0].close;
				q.sample[0] = ~loadWAV.(path); // load buffer A recording
			}
		);
	});

	elements.recB = Button(elements.w, Rect(0,0,90,30))
	.states_([["B READY", Color.green, Color.gray], ["B REC", Color.white, Color.red]])
	.action_({|obj|
		var val=obj.value;
		var path = (~path.pathOnly++"wav/bufB.wav").standardizePath;
		if(
			val==1,
			{if(y[1].isPlaying, {}, {
				q.sampleBufs[1].write(path, "wav", "float", 0, 0, true); // create file, leave open
				y[1] = Synth.tail(s, \ace_rec, [\buf, q.sampleBufs[1].bufnum, \inOffset, ~inOffset]).register; // record
			})},
			{
				y[1].free;
				q.sampleBufs[1].close;
				q.sample[1] = ~loadWAV.(path); // load buffer B recording
			}
		);
	});


	elements.recompute = Button(elements.w, Rect(0,0,100,30))
	.states_([["recompute", Color.white, Color.gray]])
	.action_({
		if(y.isPlaying, {}, {
			~recomputeDiff.();
			~recomputeWeights.();
			updatePlotMags.(elements.plotMags);
			updatePlotDiff.(elements.plotDiff);
			updatePlotWeights.(elements.plotWeights);
		});
	});

	elements.loadA = Button(elements.w, Rect(0,0,90,30))
	.states_([["load A", Color.white, Color.gray]])
	.action_({
		if(y.isPlaying, {}, {
			FileDialog({ |path|
				postln("Load A:" + path);
				q.sample[0] = ~loadWAV.(path); // load buffer A recording
			}, {}, stripResult: true);
		});
	});

	elements.loadB = Button(elements.w, Rect(0,0,90,30))
	.states_([["load B", Color.white, Color.gray]])
	.action_({
		if(y.isPlaying, {}, {
			FileDialog({ |path|
				postln("Load B:" + path);
				q.sample[1] = ~loadWAV.(path); // load buffer A recording
			}, {}, stripResult: true);
		});
	});

	/*
	var sliderExp = EZSlider(w, 550@25, "  "++"Exponent"++"  ", ControlSpec(1, 100, \exp, 0.0, p.exponent, ""), labelWidth: 100, unitWidth:30,initVal:p.exponent, numberWidth: 60, layout: \horz)
	.setColors(Color.grey, Color.white)
	.font_(Font("Helvetica", 16))
	.action_({|ez|
	p.exponent = ez.value;
	~recomputeWeights.value;
	updatePlotDiff.(plotDiff);
	updatePlotWeights.(plotWeights);
	});
	*/

	elements.sliderThresh = EZSlider(elements.w, 550@25, "  "++"Threshold"++"  ", ControlSpec(-30, 0, \lin, 0.0, p.threshold.ampdb, "dB"), labelWidth: 100, unitWidth: 30, initVal: p.threshold.ampdb, numberWidth: 60, layout: \horz)
	.setColors(Color.grey, Color.white)
	.font_(Font("Helvetica", 16))
	.action_({|ez|
		p.threshold = ez.value.dbamp;
		~recomputeWeights.();
		updatePlotDiff.(elements.plotDiff);
		updatePlotWeights.(elements.plotWeights);
		if(x.isPlaying, {x.set(\bandWeights, q.bandWeights)});
	});

	elements.sliderAtt = EZSlider(elements.w, 550@25, "  "++"Attenuation"++"  ", ControlSpec(-100, 0, \lin, 0.0, p.attBelow, "dB"), labelWidth: 100, unitWidth: 30, initVal: p.attBelow, numberWidth: 60, layout: \horz)
	.setColors(Color.grey, Color.white)
	.font_(Font("Helvetica", 16))
	.action_({|ez|
		p.attBelow = ez.value;
		~recomputeWeights.();
		updatePlotWeights.(elements.plotWeights);
		if(x.isPlaying, {x.set(\bandWeights, q.bandWeights)});
	});

	elements.headingMags = StaticText(elements.w, 550@30).string_("Magnitude Spectra A (blue) and B (red):")
	.font_(Font("Helvetica", 16));
	elements.plotMagsView = View(elements.w, Rect(0, 0, 550, 150));
	elements.plotMags = Plotter("", Rect(0, 0, 550, 150), parent: elements.plotMagsView);

	elements.headingDiff = StaticText(elements.w, 550@30).string_("Normalized difference spectrum:")
	.font_(Font("Helvetica", 16));
	elements.plotDiffView = View(elements.w, Rect(0, 0, 550, 150));
	elements.plotDiff = Plotter("", Rect(0, 0, 550, 150), parent: elements.plotDiffView);

	elements.headingWeights = StaticText(elements.w, 550@30).string_("Gammatone filter weights:")
	.font_(Font("Helvetica", 16));
	elements.plotWeightsView = View(elements.w, Rect(0, 0, 550, 150));
	elements.plotWeights = Plotter("", Rect(0, 0, 550, 150), parent: elements.plotWeightsView);


	updatePlotMags.(elements.plotMags);
	updatePlotDiff.(elements.plotDiff);
	updatePlotWeights.(elements.plotWeights);


	elements;
};


"done.".postln;