// ACE Offline Processing - Simple Test Version
// This version gives clear feedback at each step

(
"=== Starting ACE Offline Loading ===".postln;

// Check if files exist
var basePath = "/Users/luisreyes/Sonify/ace-master/sc/";
var requiredFiles = ["ace-rt_settings", "ace-rt_functions", "ace-rt_calculations", "ace-rt_synthdefs"];
var allFilesExist = true;

requiredFiles.do { |filename|
    var fullPath = basePath ++ filename ++ ".scd";
    if (File.exists(fullPath), {
        ("✓ Found: " ++ filename ++ ".scd").postln;
    }, {
        ("✗ Missing: " ++ filename ++ ".scd").postln;
        allFilesExist = false;
    });
};

if (allFilesExist, {
    "=== All required files found. Loading ACE components... ===".postln;
    
    // Load each file individually with feedback
    try {
        (basePath ++ "ace-rt_settings.scd").load;
        "✓ Settings loaded".postln;
        
        (basePath ++ "ace-rt_functions.scd").load;
        "✓ Functions loaded".postln;
        
        (basePath ++ "ace-rt_calculations.scd").load;
        "✓ Calculations loaded".postln;
        
        (basePath ++ "ace-rt_synthdefs.scd").load;
        "✓ SynthDefs loaded".postln;
        
        "=== All ACE components loaded successfully! ===".postln;
        
    } { |error|
        ("ERROR loading ACE components: " ++ error.errorString).postln;
    };
    
}, {
    "ERROR: Some required ACE files are missing!".postln;
});
)

// Define the offline processing function with better error handling
(
~aceOfflineProcessSimple = { |inputPath, outputPath, aceParams|
    var server, inputBuffer, outputBuffer, synth;
    var inputFile, numFrames, sampleRate, numChannels;
    var processTime;
    
    "=== Starting ACE Offline Processing ===".postln;
    
    // Check if input file exists
    if (File.exists(inputPath).not, {
        ("ERROR: Input file does not exist: " ++ inputPath).postln;
        ^nil;
    });
    
    "✓ Input file exists".postln;
    
    // Get file info
    inputFile = SoundFile.new;
    if (inputFile.openRead(inputPath).not, {
        ("ERROR: Cannot read input file: " ++ inputPath).postln;
        ^nil;
    });
    
    numFrames = inputFile.numFrames;
    sampleRate = inputFile.sampleRate;
    numChannels = inputFile.numChannels;
    inputFile.close;
    
    ("✓ File info - Duration: " ++ (numFrames/sampleRate).round(0.01) ++ "s, SR: " ++ sampleRate ++ ", Channels: " ++ numChannels).postln;
    
    // Ensure server is running
    server = Server.default;
    if (server.serverRunning.not, {
        "Starting SuperCollider server...".postln;
        server.boot;
        server.sync;
        "✓ Server started".postln;
    }, {
        "✓ Server already running".postln;
    });
    
    // Load input file
    "Loading input file into buffer...".postln;
    inputBuffer = Buffer.read(server, inputPath);
    server.sync;
    "✓ Input buffer loaded".postln;
    
    // Create output buffer
    outputBuffer = Buffer.alloc(server, numFrames, numChannels);
    server.sync;
    "✓ Output buffer created".postln;
    
    // Simple ACE processing SynthDef
    SynthDef(\aceOfflineSimple, {
        arg inputBuf, outputBuf, wet = 0.8, vol = 0, rho = 20, beta = 6;
        var sig, enhanced, out;
        
        // Read from buffer
        sig = PlayBuf.ar(numChannels, inputBuf, BufRateScale.kr(inputBuf), doneAction: 2);
        
        // Simple enhancement (placeholder for full ACE)
        enhanced = sig;
        enhanced = HPF.ar(enhanced, 50); // High-pass filter
        enhanced = Compander.ar(enhanced, enhanced, 0.5, 1, 0.1, 0.01, 0.1); // Simple dynamics
        enhanced = LPF.ar(enhanced, 8000); // Low-pass filter
        
        // Mix with original
        out = XFade2.ar(sig, enhanced, (wet * 2) - 1);
        out = out * vol.dbamp;
        
        // Record to output buffer
        RecordBuf.ar(out, outputBuf, doneAction: 2);
    }).load(server);
    
    server.sync;
    "✓ SynthDef loaded".postln;
    
    // Start processing
    "Starting audio processing...".postln;
    synth = Synth(\aceOfflineSimple, [
        \inputBuf, inputBuffer,
        \outputBuf, outputBuffer,
        \wet, aceParams !? aceParams[\wet] ?? 0.8,
        \vol, aceParams !? aceParams[\vol] ?? 0
    ]);
    
    // Wait for processing
    processTime = (numFrames / sampleRate) + 2;
    
    fork {
        processTime.wait;
        
        "Writing output file...".postln;
        outputBuffer.write(outputPath, "wav", "float");
        server.sync;
        
        // Cleanup
        inputBuffer.free;
        outputBuffer.free;
        
        "=== ACE Processing Complete! ===".postln;
        ("Output saved to: " ++ outputPath).postln;
    };
};

"=== ACE Offline Processor Ready ===".postln;
"Usage: ~aceOfflineProcessSimple.(\"input.wav\", \"output.wav\")".postln;
)