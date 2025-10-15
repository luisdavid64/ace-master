//
// Settings for ACE
//
"ACE: loading settings ... ".post;

p = ( // control parameters
	sigma: 3, // LI: standard deviation in ERB
	rho: 25, // LI: Exponent
	beta: 1, // EX: Exponent
	nu: -60, // TR: Threshold in dB
	dBregDelta: -96, // Regularization value in dB
	dbNoise: -96, // noise level in dB
	tauLI: 7, // LI: time constant in ms
	fHSF: 4000, // High-Shelf cutoff frequency in Hz
	sHSF: 0.1, // High-Shelf slope
	dbHSF: 0, // High-Shelf gain in dB
	fHPFtc: 4000, // TR: High-pass filter cutoff frequency in Hz
	tauAtc: 7, // TR: slow attack time constant in ms
	tauDtc: 16, // TR: slow decay time constant in ms
	mu: -3, // EX: threshold in dB
	tauAdp: 7, // DP: slow attack time constant in ms
	t60atCutoff: 0.72, // DP: Reverberation time RT60 at the cutoff frequency
	dpCutoff: 1000, // DP: cutoff frequency for decay prolongation
	tauSP: 2, // general smoothing time constant in ms
	tauEX: 7, // EX: smoothing time constant in ms
	wet: 0.9, // dry-wet balance
	dimWeight: 1, // balance between TCE and SCE
	vol: 6, // master volume in dB
	exponent: 3, // exponent for filter expansion
	threshold: 0.5, // threshold for filter expansion
	attBelow: 0, // attenuation of bands below threshold
	fcMin: 50, // lowest center frequency in Hz
	fcMax: 20000, // highest center frequency in Hz
	nBands: 60, // number of bands
	fOrder: 4, // filter order of gammatone filters
	fAtt: 4, // filter attenuation at cutoff frequencies in dB
	nBandsLI: 16, // LI: number of neighboring bands to be processed
	fHPF: 50, // High-pass filter cutoff frequency
	nfft: 4096, // fft size for STFT. Must be >= block size.
	blockSize: 2048, // block size for STFT (hann-windowed). Must be <= fft size.
	gate: 1.0 // gate for fadein and fadeout
);

p.presets = [
	["0: cardboard box", 0.075, [1,0]],
	["1: glass table", 0.84, [25,0]],
	["2: ceramic coffee cup", 0.36, [25,1]],
	["3: forest soil", 0.075, [25,9]],
	["4: plastic box", 0.36, [2,0]],
	["5: plastic bottle", 0.15, [6,0]],
	["6: wooden bar", 0.36, [25,1]],
	["7: metal box", 0.84, [25,1]],
	["8: glass bowl", 0.84, [25,9]],
	["9: metal table", 0.84, [25,1]],
	["A: high-density material", 0.84, [25,1]],
	["B: low-density material", 0.15, [25,0]]
];


"done.".postln;