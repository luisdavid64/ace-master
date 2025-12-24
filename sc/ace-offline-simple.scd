(
// ====================================================
// ACE OFFLINE / FILE RENDER — SINGLE FILE (FIXED)
// ====================================================

s = Server.local;

~path = PathName(thisProcess.nowExecutingPath);

// helper to load ACE files
~loadfiles = { |targets|
    targets.do { |t|
        (~path.pathOnly ++ "ace-rt_" ++ t ++ ".scd").load;
    }
};

// ---------- BOOT & LOAD OPTIONS ----------
~loadfiles.(\options);
~setOptions.(s, \default);

s.options.sampleRate = 44100;
s.options.numInputBusChannels = 2;
s.options.numOutputBusChannels = 2;

s.reboot;

// ---------- LOAD ACE CORE (NO GUI) ----------
~loadfiles.([\settings, \functions, \calculations, \synthdefs]);

// ---------- FAKE LIVE INPUT ----------
SynthDef(\bufferToInput, { |bufnum|
    var sig;
    sig = PlayBuf.ar(2, bufnum, doneAction: 2);
    ReplaceOut.ar(0, sig);   // inject into SoundIn
}).add;

// ====================================================
// EVERYTHING THAT WAITS GOES IN A ROUTINE
// ====================================================

s.waitForBoot({

    "Server booted, scheduling ACE render".postln;

    ~loadfiles.([\settings, \functions, \calculations, \synthdefs]);

    SynthDef(\bufferToInput, { |bufnum|
        var sig;
        sig = PlayBuf.ar(2, bufnum, doneAction: 2);
        ReplaceOut.ar(0, sig);
    }).add;

    // 🔑 defer routine start slightly
    AppClock.sched(0.1, {

        Routine({

            "ACE routine started".postln;

            ~buf = Buffer.read(
                s,
                "/Users/luisreyes/Sonify/SonifyOCT/utilities/runs/man_tracking_inference_22_12_2025_11_31_33/recording.wav"
            );

            s.sync;

            s.prepareForRecord;
            s.record;

            x = Synth.tail(s, \ace,
                p.getPairs ++ [
                    \gate, 1,
                    \out, 0
                ]
            );

            ~player = Synth.head(s, \bufferToInput, [
                \bufnum, ~buf
            ]);

            (~buf.duration + 0.5).wait;

            s.stopRecording;
            "ACE render finished".postln;

        }).play;

        nil; // required for AppClock
    });

});


)
