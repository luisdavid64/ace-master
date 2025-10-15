
// 1. Load the offline processor
//"/ace-offline-complete.scd".load
s.boot
s.meter;
s.plotTree;
s.reboot;
s.quit;


~sound=Buffer.read(s,"/Users/luisreyes/Desktop/samples/audio1.wav");
~sound.play;

// 2 Load the scripts for ACE
"/Users/luisreyes/Sonify/ace-master/sc/ace-offline-complete.scd".load

// 3. Process sampleA.wav with default settings
fork {
	~aceOfflineProcess.(
		"/Users/luisreyes/Desktop/samples/audio1.wav",
		"/Users/luisreyes/Desktop/ace_audio1.wav"
	)
};

// 4. Process with custom settings

~aceParams = (
	dBregDelta: -96,
	fHPF: 50,
	fHSF: 12000,
	sHSF: 0.1,
	dbHSF: 0,
	fHPFtc: 4000,
	tauAtc: 5,
	tauDtc: 7,
	nu: -40,
	dbNoise: -96,
	tauLI: 7,
	beta: 8,
	mu: -5,
	rho: 15,
	tauAdp: 7,
	t60atCutoff: 0.8,
	dpCutoff: 1000.0,
	tauSP: 2,
	dimWeight: 0.5,
	wet: 1,
	vol: 0,
	tauEX: 7,
	lagt60: 0.1
);

// usage

//fork {
//	~aceOfflineProcess.(
//		input_path,
//		output_path,
//		ace_parameters
//	)
};

fork {
	~aceOfflineProcess.(
		"/Users/luisreyes/Desktop/samples/audio1.wav",
		"/Users/luisreyes/Desktop/ace_audio1.wav",
		~aceParams
	)
};