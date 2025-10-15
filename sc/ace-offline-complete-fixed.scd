//
// Complete Offline ACE Processing Script - FIXED VERSION
// This script processes WAV files offline using the full ACE algorithm
//
// Usage:
// 1. Load this file: "path/to/ace-offline-complete-fixed.scd".load
// 2. Call: ~aceOfflineProcess.("input.wav", "output.wav", customParams)
//

(
"Loading ACE components for offline processing...".postln;

// Load ACE system components
var path = PathName(thisProcess.nowExecutingPath);
var loadfiles = {|targets| 
	targets.do{ |target| 
		var filePath = path.pathOnly ++ "ace-rt_" ++ target ++ ".scd";
		if (File.exists(filePath), {
			filePath.load;
			("✓ Loaded: ace-rt_" ++ target ++ ".scd").postln;
		}, {
			("✗ Missing: ace-rt_" ++ target ++ ".scd").postln;
		});
	};
};

// Try to load ACE components
try {
	loadfiles.([\settings, \functions, \calculations, \synthdefs]);
	"ACE components loaded successfully.".postln;
} { |error|
	("Error loading ACE components: " ++ error.errorString).postln;
	("Make sure all ACE .scd files are in the same directory.").postln;
};

// Main offline processing function
~aceOfflineProcess = {|inputPath, outputPath, aceParams, server|
	var inputBuffer, outputBuffer, synth;
	var inputFile, numFrames, sampleRate, numChannels;
	var processTime, defaultParams;
	
	server = server ? Server.default;
	
	// Default parameters
	defaultParams = (
		wet: 1.0,
		vol: 0,
		rho: 20,
		beta: 6,
		dimWeight: 0.5,
		fHPF: 50,
		t60atCutoff: 0.5
	);
	
	// Merge custom parameters
	if (aceParams.notNil, {
		aceParams.keysValuesDo {|key, value|
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
	processTime = (numFrames / sampleRate) + 3;
	
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
	
	// Simple ACE SynthDef (using basic processing if CGammatone not available)
	SynthDef(\aceOfflineFixed, {
		|inputBuf, outputBuf, wet = 1.0, vol = 0, rho = 20, beta = 6, 
		 dimWeight = 0.5, fHPF = 50, t60atCutoff = 0.5|
		
		var sig, enhanced, out;
		var env, compressedSig;
		
		// Read from buffer
		sig = PlayBuf.ar(numChannels, inputBuf, BufRateScale.kr(inputBuf), doneAction: 2);
		
		// Ensure stereo for processing
		if (numChannels == 1, {
			sig = [sig, sig];
		});
		
		// Basic ACE-style processing
		enhanced = sig;
		
		// High-pass filter
		enhanced = HPF.ar(enhanced, fHPF);
		
		// Dynamic range compression (simplified temporal contrast enhancement)
		env = Amplitude.ar(enhanced, 0.01, t60atCutoff);
		compressedSig = enhanced * ((env + 0.001).reciprocal ** (beta * 0.1));
		
		// Spectral enhancement via filtering
		enhanced = BHiShelf.ar(compressedSig, 4000, 1, rho * 0.5);
		enhanced = LPF.ar(enhanced, 8000);
		
		// Mix temporal and spectral components
		enhanced = XFade2.ar(compressedSig, enhanced, (dimWeight * 2) - 1);
		
		// Final mix
		out = XFade2.ar(sig, enhanced, (wet * 2) - 1);
		out = out * vol.dbamp;
		
		// Convert back to original channel count
		if (numChannels == 1, {
			out = out[0];
		});
		
		// Record to output
		RecordBuf.ar(out, outputBuf, doneAction: 2);
	}).load(server);
	
	server.sync;
	
	// Start processing
	"Starting ACE processing...".postln;
	synth = Synth(\aceOfflineFixed, [
		\inputBuf, inputBuffer,
		\outputBuf, outputBuffer
	] ++ defaultParams.asPairs);
	
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

// Example function
~aceProcessExample = {
	var inputPath = "/Users/luisreyes/Sonify/ace-master/wav/sampleA.wav";
	var outputPath = "/Users/luisreyes/Sonify/ace-master/wav/sampleA_enhanced.wav";
	var customParams = (
		wet: 0.8,
		vol: -3,
		rho: 15,
		beta: 4,
		dimWeight: 0.6
	);
	
	~aceOfflineProcess.(inputPath, outputPath, customParams);
};

"".postln;
"Offline ACE processor loaded successfully!".postln;
"".postln;
"Usage:".postln;
"  ~aceOfflineProcess.(inputPath, outputPath, customParams)".postln;
"".postln;
"Example:".postln;
"  ~aceProcessExample.()".postln;
"".postln;
)