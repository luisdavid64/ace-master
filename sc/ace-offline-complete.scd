(
// Load ACE system components
var path = PathName(thisProcess.nowExecutingPath);
var loadfiles = {|targets|
	targets.do{ |target|
		(path.pathOnly ++ path.fileNameWithoutExtension.replace("offline-complete", "") ++ "_" ++ target ++ ".scd").load
	}
};

// Load required ACE components
"Loading ACE components for offline processing...".postln;

try {
	// Load settings, functions, calculations first
	loadfiles.([\settings, \functions, \calculations]);

	// Wait a moment for loading
	0.1.wait;

	// Load synthdefs
	loadfiles.([\synthdefs]);

	"ACE components loaded successfully.".postln;
} { |error|
	("Error loading ACE components: " ++ error.errorString).postln;
	("Make sure all ACE .scd files are in the same directory and ACEUGens are installed.").postln;
};

// Main offline processing function
~aceOfflineProcess = { |inputPath, outputPath, aceParams, server|
	var inputBuffer, outputBuffer, synth;
	var inputFile, numFrames, sampleRate, numChannels;
	var processTime;
	var defaultParams;

	server = server ? Server.default;

	// Default parameters - these will be overridden by loaded settings
	defaultParams = ();
	if (p.notNil, {
		// Use loaded ACE parameters
		defaultParams = p.copy;
	}, {
		// Fallback parameters if p is not loaded
		defaultParams = (
			dBregDelta: -96,
			fHPF: 50,
			fHSF: 8000,
			sHSF: 0.1,
			dbHSF: 0,
			fHPFtc: 4000,
			tauAtc: 5,
			tauDtc: 7,
			nu: -40,
			dbNoise: -96,
			tauLI: 7,
			beta: 8,
			mu: -3,
			rho: 30,
			tauAdp: 7,
			t60atCutoff: 0.5,
			dpCutoff: 1000.0,
			tauSP: 2,
			dimWeight: 0.5,
			wet: 1.0,
			vol: 0,
			tauEX: 7,
			lagt60: 0.1
		);
	}); // ← FIX 1: removed stray "};" (this line properly closes the if(...){...}{...}); block

	// Merge custom parameters
	if (aceParams.notNil, {
		aceParams.keysValuesDo { |key, value|
			defaultParams[key] = value;
		};
	});

	// Check input file
	if (File.exists(inputPath).not, {
		("Error: Input file does not exist: " ++ inputPath).error;
		^nil;
	});

	// Get file information
	inputFile = SoundFile.new;
	if (inputFile.openRead(inputPath).not, {
		("Error: Cannot read input file: " ++ inputPath).error;
		^nil;
	});

	numFrames = inputFile.numFrames;
	sampleRate = inputFile.sampleRate;
	numChannels = inputFile.numChannels;
	inputFile.close;

	("Processing file: " ++ inputPath).postln;
	("Duration: " ++ (numFrames / sampleRate).round(0.01) ++ " seconds").postln;
	("Sample rate: " ++ sampleRate ++ " Hz").postln;
	("Channels: " ++ numChannels).postln;

	// Calculate processing time
	processTime = (numFrames / sampleRate) + 3; // Add 3 seconds buffer

	// Ensure server is running
	if (server.serverRunning.not, {
		"Starting SuperCollider server...".postln;
		server.boot;
		server.sync;
	});

	// Load buffers
	"Loading audio file...".postln;
	inputBuffer = Buffer.read(server, inputPath);
	server.sync;

	outputBuffer = Buffer.alloc(server, numFrames, numChannels);
	server.sync;

	// Create offline version of ACE SynthDef
	SynthDef(\aceOfflineComplete, {
		arg inputBuf, outputBuf,
		// All ACE parameters with defaults
		dBregDelta = -96,
		fHPF = 50,
		fHSF = 8000,
		sHSF = 0.1,
		dbHSF = 0,
		fHPFtc = 4000,
		tauAtc = 5,
		tauDtc = 7,
		nu = -40,
		dbNoise = -96,
		tauLI = 7,
		beta = 8,
		mu = -3,
		rho = 30,
		tauAdp = 7,
		t60atCutoff = 0.5,
		dpCutoff = 1000.0,
		tauSP = 2,
		dimWeight = 0.5,
		wet = 1,
		vol = 0,
		tauEX = 7,
		lagt60 = 0.1;

		var sig, sigTC, sigSC, envTCe, ck, env0, env, envS, envSmax, sigACE, out;
		var logk, exT60, tcT60D, tcT60A, spT60, liT60, dpT60A;
		var nBands, nBandsLI, fSigns, fcs, bws, gammas, dpT60, regDelta, rsHSF;
		var bandWeights;

		// Use loaded parameters if available
		bandWeights = if (p.notNil and: {p.nBands.notNil}, {
			1.0!p.nBands;
		}, {
			1.0!32; // Fallback
		});

		nBands = if (p.notNil and: {p.nBands.notNil}, { p.nBands }, { 32 });
		nBandsLI = if (p.notNil and: {p.nBandsLI.notNil}, { p.nBandsLI }, { 3 });

		// Use calculated parameters if available
		fSigns = if (q.notNil and: {q.fSigns.notNil}, { q.fSigns }, { 1.0!nBands });
		fcs = if (q.notNil and: {q.fc.notNil}, { q.fc }, { Array.geom(nBands, 80, 1.23) });
		bws = if (q.notNil and: {q.bw.notNil}, { q.bw }, { fcs * 0.25 });
		gammas = if (q.notNil and: {q.gammas.notNil}, { q.gammas }, { {1.0!nBands}!nBands });

		// Calculated values
		logk = log(1000)/1000;
		exT60 = tauEX*logk;
		tcT60D = tauDtc*logk;
		tcT60A = tauAtc*logk;
		spT60 = tauSP*logk;
		liT60 = tauLI*logk;
		dpT60A = tauAdp*logk;
		regDelta = dBregDelta.dbamp;
		rsHSF = sHSF.reciprocal;
		dpT60 = (fcs.reciprocal*dpCutoff*t60atCutoff).clip(0, t60atCutoff);

		// Read from input buffer
		sig = PlayBuf.ar(numChannels, inputBuf, BufRateScale.kr(inputBuf), doneAction: 2);

		// Ensure stereo for processing
		if (numChannels == 1, { sig = [sig, sig] });

		// High-pass filter
		sig = HPF.ar(sig, fHPF);

		//// TCE (Temporal Contrast Enhancement)
		sigTC = HPF.ar(sig, fHPFtc);
		envTCe = Amplitude.ar(sigTC.abs, 0, tcT60D);
		envTCe = (envTCe - Amplitude.ar(envTCe, tcT60A, 0) - nu.dbamp).clip(0, 1);
		sigTC = sigTC * (envTCe / (Amplitude.ar(envTCe, 0, tcT60D) + regDelta));

		//// SCE (Spectral Contrast Enhancement)
		sigSC = BHiShelf.ar(sig, fHSF, rsHSF, dbHSF);
		sigSC = sigSC + (PinkNoise.ar(16!2).fold(-1,1) * dbNoise.dbamp); // ← FIX 2: use dbNoise.dbamp

		// Try to use CGammatone if available, otherwise use simplified version
		ck = if (CGammatone.respondsTo(\ar), {
			// Full ACE processing with CGammatone filterbank
			sigSC.collect{|item,i| CGammatone.ar(item, fcs, bws).flop};
		}, {
			// Simplified version without CGammatone
			("Warning: CGammatone UGen not available. Using simplified filterbank.").postln;
			sigSC.collect{|item,i|
				fcs.collect{|fc, band|
					var filtered = BPF.ar(item, fc, fc/bws[band]);
					[filtered, filtered * 0]; // Fake complex output [real, imag]
				}.flop;
			};
		});

		// Process envelope
		env0 = 2.collect{|i| ck[i].squared.sum.sqrt.clip(0,1)};
		env = ck.sum.madd(0.5).squared.sum.sqrt.clip(0,1); // (madd usage is unchanged)

		// LI (Lateral Inhibition)
		envS = Amplitude.ar(env, liT60, liT60);
		env = env * ((envS / (nBands.collect{|i|
			var iRange=(0.max(i-nBandsLI)..(nBands-1).min(i+nBandsLI));
			(envS[iRange].squared*gammas[i][iRange].normalizeSum).sum;
		}.sqrt + regDelta))**rho).clip(0, 1);

		// EX (Expansion)
		envS = Amplitude.ar(env, exT60, exT60);
		envSmax = envS.max; // FIX: ArrayMax doesn't exist, use .max instead
		env = env * ((envS/(mu.dbamp*envSmax + regDelta))**beta).clip(0, envSmax/(envS+regDelta));

		// DP (Decay Processing)
		envS = Amplitude.ar(env, dpT60A, 0);
		env = (env-envS) + nBands.collect{|i| Amplitude.ar(envS[i], 0, dpT60[i]);};

		// Summation
		env0 = 2.collect{|i| Amplitude.ar(env0[i], spT60, spT60)};
		env = Amplitude.ar(env, spT60, spT60);
		sigACE = 2.collect{|i| (ck[i][0] * fSigns * env * bandWeights / (env0[i] + regDelta)).sum};
		sigACE = BHiShelf.ar(sigACE, fHSF, rsHSF, dbHSF.neg);
		sigACE = XFade2.ar(sigTC, sigACE, (dimWeight*2)-1);
		sigACE = HPF.ar(sigACE, fHPF);

		// Final mix
		out = XFade2.ar(sig, sigACE, (wet*2)-1);
		out = (out * vol.dbamp).clip(-1,1);

		// Convert back to original channel configuration
		if (numChannels == 1, { out = out[0] });

		// Record to output buffer
		RecordBuf.ar(out, outputBuf, doneAction: 2);
	}).load(server);

	server.sync;

	// Start processing synth
	"Starting ACE processing...".postln;
	synth = Synth(\aceOfflineComplete,
		[\inputBuf, inputBuffer, \outputBuf, outputBuffer] ++ defaultParams.asPairs
	);

	// Wait for processing to complete
	fork {
		processTime.wait;

		// Write output file
		"Writing output file...".postln;
		outputBuffer.write(outputPath, "wav", "float");
		server.sync;

		// Cleanup
		inputBuffer.free;
		outputBuffer.free;

		("ACE processing complete!").postln;
		("Enhanced audio saved to: " ++ outputPath).postln;
		("Original duration: " ++ (numFrames/sampleRate).round(0.01) ++ "s").postln;
	};
};

// Convenience function with example parameters
~aceProcessExample = {
	var inputPath = "/Users/luisreyes/Sonify/ace-master/sc/wav/sampleB.wav";
	var outputPath = "/Users/luisreyes/Sonify/ace-master/sc/wav/sampleB_enhanced.wav";
	var customParams = (
		wet: 0.8,     // 80% ACE processing, 20% original
		vol: -3,      // -3dB output level
		rho: 25,      // Spectral contrast strength
		beta: 6,      // Temporal contrast strength
		dimWeight: 0.6 // Balance between spectral and temporal
	);

	~aceOfflineProcess.(inputPath, outputPath);
};

"".postln;
"Offline ACE processor loaded successfully!".postln;
"".postln;
"Usage:".postln;
"  ~aceOfflineProcess.(inputPath, outputPath, customParams)".postln;
"".postln;
"Example:".postln;
"  ~aceProcessExample.() // Process sampleA.wav with example settings".postln;
"".postln;
"Custom example:".postln;
"  ~aceOfflineProcess.(\"input.wav\", \"output.wav\", (wet: 0.5, vol: -6))".postln;
"".postln;
)
