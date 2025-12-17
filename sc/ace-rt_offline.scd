
(
// to evaluate, click here and hit Ctrl+Return
// this may take a while... (see post window for status)
~path = PathName(thisProcess.nowExecutingPath); // current path
~loadfiles = {|targets| targets.do{ |target| (~path.pathOnly ++ "ace-rt" ++ "_" ++ target ++ ".scd").load } }; // function
~loadfiles.([\options]); // load functions and settings
~setOptions.(s, \default); // set server options for specific audio interface as 2nd argument
~loadfiles.([\settings, \functions, \calculations, \synthdefs, \gui, \ctl]); // load functions and settings
[p.notNil, q.notNil].postln;
~loadfiles.([\nrt])

)

s.boot;

SynthDef(\pb_test, { |bufnum=0|
    var sig;
    sig = PlayBuf.ar(2, bufnum, BufRateScale.kr(bufnum), doneAction:2);
    Out.ar(0, sig * 0.5);
}).writeDefFile("/Users/luisreyes/Desktop/".standardizePath);

/*(
Buffer.read(
    s,
    "/Users/luisreyes/Desktop/samples/audio1.wav",
    action: { |buf|
        buf.postln;                 // now it has frames/channels
        SynthDef(\pb_test_rt, {
            var sig;
            sig = PlayBuf.ar(1, buf.bufnum, doneAction: 2);
            Out.ar(0, sig ! 2 * 0.5);
        }).add;

        Synth(\pb_test_rt);
    }
);
)*/

(
var inPath = "/Users/luisreyes/Desktop/samples/audio1.wav".standardizePath;
var outPath = "/Users/luisreyes/Desktop/test_render.wav".standardizePath;
//var defPath = "/Users/luisreyes/Desktop/ace_nrt.scsyndef".standardizePath;
var defPath = "/Users/luisreyes/Sonify/ace-master/sc/ace_nrt.scsyndef".standardizePath;

var sf = SoundFile.openRead(inPath);
var dur = sf.numFrames / sf.sampleRate;

var options = ServerOptions.new
    .numOutputBusChannels_(2)
    .numInputBusChannels_(0)
    .sampleRate_(44100);

var score = Score([
    [0.0, [\d_load, defPath]],
    [0.0, [\b_allocRead, 0, inPath]],
    [0.05, [\s_new, \ace_nrt, 1000, 0, 0, \bufnum, 0]],
    [dur + 0.2, [\quit]]
]);

score.recordNRT(
    outputFilePath: outPath,
    headerFormat: "wav",
    sampleFormat: "int16",
    options: options,
    duration: dur + 0.3
);

("Rendered to: " ++ outPath).postln;
)
