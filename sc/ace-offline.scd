//
// Offline ACE (Auditory Contrast Enhancement) processing script
// Process WAV files offline and save enhanced output
// Based on the real-time ACE implementation
//

(
// Function to process a WAV file with ACE offline
~processACEOffline = { |inputPath, outputPath, aceParams|
	var server, buffer, outputBuffer, synthDef, synth;
	var inputFile, outputFile;
	var numFrames, sampleRate, numChannels;
	var processTime;
	
	// Default ACE parameters (can be overridden)
	aceParams = aceParams ?? {
		Dictionary.newFrom([
			\dBregDelta, -96,
			\fHPF, 50,
			\fHSF, 8000,
			\sHSF, 0.1,
			\dbHSF, 0,
			\fHPFtc, 4000,
			\tauAtc, 5,
			\tauDtc, 7,
			\nu, -40,
			\dbNoise, -96,
			\tauLI, 7,
			\beta, 8,
			\mu, -3,
			\rho, 30,
			\tauAdp, 7,
			\t60atCutoff, 0.5,
			\dpCutoff, 1000.0,
			\tauSP, 2,
			\dimWeight, 0.5,
			\wet, 1.0,
			\vol, 0,
			\tauEX, 7,
			\lagt60, 0.1
		]);
	};
	
	// Check if input file exists
	if (File.exists(inputPath).not) {
		("Input file does not exist: " ++ inputPath).postln;
		^nil;
	};
	
	// Get file info
	inputFile = SoundFile.new;
	inputFile.openRead(inputPath);
	numFrames = inputFile.numFrames;
	sampleRate = inputFile.sampleRate;
	numChannels = inputFile.numChannels;
	inputFile.close;
	
	("Processing: " ++ inputPath).postln;
	("Duration: " ++ (numFrames / sampleRate) ++ " seconds").postln;
	("Sample Rate: " ++ sampleRate).postln;
	("Channels: " ++ numChannels).postln;
	
	// Calculate processing time (add some buffer)
	processTime = (numFrames / sampleRate) + 2;
	
	// Start server if not running
	if (Server.default.serverRunning.not) {
		Server.default.boot;
		Server.default.sync;
	};
	
	server = Server.default;
	
	// Load input file into buffer
	buffer = Buffer.read(server, inputPath);
	server.sync;
	
	// Create output buffer for recording
	outputBuffer = Buffer.alloc(server, numFrames, numChannels);
	server.sync;
	
	// Define offline ACE SynthDef
	SynthDef(\aceOffline, {
		arg buf, outBuf,
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
		
		// Read from buffer (supports mono and stereo)
		sig = PlayBuf.ar(numChannels, buf, BufRateScale.kr(buf), doneAction: 2);
		if (numChannels == 1) { sig = [sig, sig] }; // Convert mono to stereo for processing
		
		// Initialize parameters (simplified version - you'll need to load the full parameter sets)
		logk = log(1000)/1000;
		exT60 = tauEX*logk;
		tcT60D = tauDtc*logk;
		tcT60A = tauAtc*logk;
		spT60 = tauSP*logk;
		liT60 = tauLI*logk;
		dpT60A = tauAdp*logk;
		regDelta = dBregDelta.dbamp;
		rsHSF = sHSF.reciprocal;
		
		// Simplified filterbank parameters (you'll need to load the full sets from calculations)
		nBands = 32; // Default number of bands
		fcs = Array.geom(nBands, 80, 1.23); // Center frequencies (approximate)
		bws = fcs * 0.25; // Bandwidths (approximate)
		dpT60 = (fcs.reciprocal*dpCutoff*t60atCutoff).clip(0, t60atCutoff);
		bandWeights = 1.0!nBands;
		
		// High-pass filter
		sig = HPF.ar(sig, fHPF);
		
		// TCE (Temporal Contrast Enhancement)
		sigTC = HPF.ar(sig, fHPFtc);
		envTCe = Amplitude.ar(sigTC.abs, 0, tcT60D);
		envTCe = (envTCe - Amplitude.ar(envTCe, tcT60A, 0) - nu.dbamp).clip(0, 1);
		sigTC = sigTC * (envTCe / (Amplitude.ar(envTCe, 0, tcT60D) + regDelta));
		
		// SCE (Spectral Contrast Enhancement) - Simplified version
		sigSC = BHiShelf.ar(sig, fHSF, rsHSF, dbHSF);
		sigSC = sigSC + (PinkNoise.ar(16!2).fold(-1,1)*dbamp(dbNoise));
		
		// Simplified filterbank processing (without CGammatone - using regular filters)
		// Note: For full ACE processing, you need the CGammatone UGen compiled and installed
		env = Amplitude.ar(sigSC.sum, liT60, liT60);
		
		// Apply contrast enhancement (simplified)
		sigACE = sigSC * ((env / (env + regDelta))**rho).clip(0, 1);
		sigACE = BHiShelf.ar(sigACE, fHSF, rsHSF, dbHSF.neg);
		sigACE = XFade2.ar(sigTC, sigACE, (dimWeight*2)-1);
		sigACE = HPF.ar(sigACE, fHPF);
		
		// Dry/wet mix and output
		out = XFade2.ar(sig, sigACE, (wet*2)-1);
		out = (out * vol.dbamp).clip(-1,1);
		
		// Convert back to original channel count
		if (numChannels == 1) { out = out[0] };
		
		// Record to output buffer
		RecordBuf.ar(out, outBuf, doneAction: 2);
	}).load(server);
	
	server.sync;
	
	// Start processing
	("Starting ACE processing...").postln;
	synth = Synth(\aceOffline, [
		\buf, buffer,
		\outBuf, outputBuffer
	] ++ aceParams.asPairs);
	
	// Wait for processing to complete
	(processTime + 1).wait;
	
	// Write output buffer to file
	outputBuffer.write(outputPath, "wav", "float");
	server.sync;
	
	// Cleanup
	buffer.free;
	outputBuffer.free;
	
	("ACE processing complete. Output saved to: " ++ outputPath).postln;
};

// Example usage function
~exampleUsage = {
	var inputPath, outputPath, aceParams;
	
	// Example paths (adjust these to your actual files)
	inputPath = "~/Desktop/input.wav".standardizePath;
	outputPath = "~/Desktop/output_ace.wav".standardizePath;
	
	// Custom ACE parameters (optional)
	aceParams = Dictionary.newFrom([
		\wet, 0.8,      // 80% processed signal, 20% dry
		\vol, -6,       // -6dB output level
		\rho, 20,       // Spectral contrast parameter
		\beta, 6,       // Temporal contrast parameter
		\dimWeight, 0.7 // More spectral than temporal enhancement
	]);
	
	// Process the file
	~processACEOffline.(inputPath, outputPath, aceParams);
};

("Offline ACE processor loaded.").postln;
("Usage: ~processACEOffline.(inputPath, outputPath, aceParams)").postln;
("Example: ~exampleUsage.()").postln;
)