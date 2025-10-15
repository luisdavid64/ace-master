//
// SynthDefs for ACE
//

"ACE: loading SynthDefs ... ".post;


// Recorder for inter-stimulus ace
SynthDef(\ace_rec, {|buf=0, inOffset=0|
	DiskOut.ar(buf, SoundIn.ar([0,1]+inOffset).sum);
}).load(s);

// Recorder for measurements, experiments, etc.
SynthDef(\ace_testrec, {|buf=1, inOffset=0, outOffset=0|
	DiskOut.ar(buf, SoundIn.ar([0,1]+inOffset)++In.ar(0+outOffset,2));
}).load(s);


// ACE Filter real-time
SynthDef(\ace, {
	arg // arguments
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
	dimWeight = 0.5, // spectral vs temporal
	wet = 1, // dry-wet
	vol = 0,
	tauEX = 7,
	lagt60 = 0.1;


	var // variables
	bandWeights = \bandWeights.kr(1.0!p.nBands, lagt60!p.nBands).clip(0, 1),
	logk = log(1000)/1000,
	exT60 = tauEX*logk,
	tcT60D = tauDtc*logk,
	tcT60A = tauAtc*logk,
	spT60 = tauSP*logk,
	liT60 = tauLI*logk,
	dpT60A = tauAdp*logk,
	nBands = p.nBands,
	nBandsLI = p.nBandsLI,
	fSigns = q.fSigns,
	fcs = q.fc,
	bws = q.bw,
	inOffset = ~inOffset,
	outOffset = ~outOffset,
	gammas = q.gammas,
	dpT60 = (fcs.reciprocal*dpCutoff*t60atCutoff).clip(0, t60atCutoff), // decay times clipped below dpCutoff
	regDelta = dBregDelta.dbamp,
	rsHSF = sHSF.reciprocal,
	sig,
	sigTC,
	sigSC,
	envTCe,
	ck,
	env0,
	env,
	envS,
	envSmax,
	sigACE,
	envState,
	gate = \gate.kr(1, 0),
	out;

	envState = Linen.kr(gate, 0.05, 1.0, 0.01, doneAction: 2); // fade with on/off switch

	sig = HPF.ar(SoundIn.ar([0,1]+~inOffset), fHPF); // high-pass filtered input signal

	//// TCE
	sigTC = HPF.ar(sig, fHPFtc); // high-pass filter
	envTCe = Amplitude.ar(sigTC.abs, 0, tcT60D); // slow decay envelope
	envTCe = (envTCe - Amplitude.ar(envTCe, tcT60A, 0) - nu.dbamp).clip(0, 1); // slow attack envelope
	sigTC = sigTC * ( envTCe / (Amplitude.ar(envTCe, 0, tcT60D) + regDelta) ); // transient signal

	//// SCE
	sigSC = BHiShelf.ar(sig, fHSF, rsHSF, dbHSF); // highshelf
	sigSC = sigSC + (PinkNoise.ar(16!2).fold(-1,1)*dbamp(dbNoise)); // add stereo pink noise (+16 because PinkNoise RMS is -16dB)
	//ck = sigSC.collect{|item,i| CGammatone.ar(item, fcs, bws)}; // filterbank, mono version
	ck = sigSC.collect{|item,i| CGammatone.ar(item, fcs, bws).flop}; // filterbank, stereo version
	// dimensions of ck: [channel][real/imag][band]
	env0 = 2.collect{|i| ck[i].squared.sum.sqrt.clip(0,1)}; // for both channels: envelope = sqrt(Re^2 + Im^2)
	env = ck.sum.madd(0.5).squared.sum.sqrt.clip(0,1); // mono mix. then: envelope = sqrt(Re^2 + Im^2)
	env = env * envState; // fade in and out with state switch

	// LI
	envS = Amplitude.ar(env, liT60, liT60); // smoothing by leaky integrator
	env = env * ((envS / (nBands.collect{|i| var iRange=(0.max(i-nBandsLI)..(nBands-1).min(i+nBandsLI)); (envS[iRange].squared*gammas[i][iRange].normalizeSum).sum;}.sqrt + regDelta))**rho).clip(0, 1);

	// EX
	envS = Amplitude.ar(env, exT60, exT60); // smoothing by leaky integrator
	envSmax = ArrayMax.ar(envS)[0]; // maximum of all smoothed envelopes
	env = env * ((envS/(mu.dbamp*envSmax + regDelta))**beta).clip(0, envSmax/(envS+regDelta));

	//// DP
	envS = Amplitude.ar(env, dpT60A, 0); // slow attack envelope
	env = (env-envS) + nBands.collect{|i| Amplitude.ar(envS[i], 0, dpT60[i]);}; // attack part + slow decay envelope

	//// Summation
	env0 = 2.collect{|i| Amplitude.ar(env0[i], spT60, spT60)}; // smoothing of original envelpe
	env = Amplitude.ar(env, spT60, spT60); // smoothing of new envelope
	sigACE = 2.collect{|i| ( ck[i][0] * fSigns  * env * bandWeights / (env0[i] + regDelta) ).sum}; // summation etc.
	sigACE = BHiShelf.ar(sigACE, fHSF, rsHSF, dbHSF.neg); // high-shelf filter
	sigACE = XFade2.ar(sigTC, sigACE, (dimWeight*2)-1); // crossfade between SCE and TCE
	sigACE = HPF.ar(sigACE, fHPF);


	out = XFade2.ar(sig, sigACE, (wet*2)-1); // volume and dry/wet control
	// out = Compander.ar(out, out, limitThresh, limitSlopeBelow, limitSlopeAbove, limitClampTime, limitRelaxTime); // Limiter

	Out.ar(outOffset, (out * envState * vol.dbamp).clip(-1,1)); // output, with state envelope, master vol, and clipping.
	// Out.ar( [130,131], 0.1 * out); // recording output (MADI FX, for demo recordings)
}).load(s);


"done.".postln;
