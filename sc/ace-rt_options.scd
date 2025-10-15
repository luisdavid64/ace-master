//
// server options for ACE
//
"ACE: loading options ... ".post;


// general options
s.options.numWireBufs = 4096;
s.options.memSize = 8192*4;
s.options.sampleRate = 48000;
s.options.blockSize = 64;


~setOptions = {|s, preset|
	switch(preset,
		\madifx, { // RME MADI FX on Linux
			~inOffset = 130;
			~outOffset = 194;
			s.options.blockSize = 32;
			s.options.numInputBusChannels = 194;
			s.options.numOutputBusChannels = 196;
		},
		\raydat, { // RME RayDat on Linux
			~inOffset = 0;
			~outOffset = 0;
			s.options.blockSize = 32;
			s.options.numInputBusChannels = 2;
			s.options.numOutputBusChannels = 2;
		},
		\fireface, { // RME Fireface UC(X) / Babyface Pro on Windows
			~inOffset = 0;
			~outOffset = 0;
			s.options.device = "ASIO : ASIO Fireface USB";
			s.options.blockSize = 128;
			s.options.hardwareBufferSize = 128;
			s.options.numInputBusChannels = 12;
			s.options.numOutputBusChannels = 12;
		},
		\maxio, { // ESI MaXiO on Windows
			~inOffset = 0;
			~outOffset = 0;
			s.options.blockSize = 64;
			s.options.numInputBusChannels = 2;
			s.options.numOutputBusChannels = 2;
			s.options.device = "ASIO 2.0 - MaXiO";
		},
		\quantum, { // Presonus Quantum 2 on Windows
			~inOffset = 0;
			~outOffset = 0;
			s.options.device = "ASIO : Quantum ASIO driver";
			s.options.numInputBusChannels = 2;
			s.options.numOutputBusChannels = 2;
			s.options.hardwareBufferSize = 64;
		},
		\steinberg, { // Steinberg UR22 mkII on Windows
			~inOffset = 0;
			~outOffset = 0;
			s.options.device = "ASIO : Yamaha Steinberg USB ASIO";
			s.options.numInputBusChannels = 2;
			s.options.numOutputBusChannels = 2;
			s.options.hardwareBufferSize = 64;
		},
		\default, { // any stereo audio interface
			~inOffset = 0;
			~outOffset = 0;
			s.options.numInputBusChannels = 2;
			s.options.numOutputBusChannels = 2;
		};
	);
	("ACE: options set for "++preset++" audio interface.").postln;
};

"done.".postln;