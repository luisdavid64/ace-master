//
// Calculations for ACE
//

"ACE: performing calculations ... ".post;


q = (); // create new dictionary for calculated values


// function to recompute the difference spectrum
~recomputeDiff = {
	q.sampleMags = [~signal2magLT.(q.sample[0], p.nfft, p.blockSize),
		~signal2magLT.(q.sample[1], p.nfft, p.blockSize)]; // magnitude spectra of A and B
	// q.magDiff = ((q.sampleMags[0]/q.sampleMags[1])-1).abs.normalize; // divide, mirror negative to positive, normalize
	// --> 1 if same, >1 if different.
	q.magDiffdb = (q.sampleMags[0].ampdb - q.sampleMags[1].ampdb).abs; // absolute magnitude difference in dB
	q.magDiff = q.magDiffdb.dbamp; // absolute magnitude difference
	q.magDiffdbMax = q.magDiffdb[1..].maxItem; // maximum difference in dB
};

// function to recompute the band weights
~recomputeWeights = {
	/*
	q.magACE = ((q.magDiff/p.threshold)**p.exponent).clip(0,1); // nonlinear ace filter
	q.bandWeights = q.magResponse.collect{ |magBand,i| // compute Gammatone filter weights
	(magBand*q.magACE).sum / magBand.sum;
	}.normalize; // normalized between 0 and 1
	*/
	// q.magACE = ((q.magDiff/p.threshold)>=1); // decide if mag difference is >= threshold
	q.magACE = (q.magDiffdb >= (q.magDiffdbMax + p.threshold.ampdb));
	q.bandWeights = Array.fill(p.nBands, p.attBelow.dbamp); // init band weights with attenuation parameter
	q.magACE.do{|item,i| item.if({q.bandWeights[q.binBands[i]] = 1.0})}; // set active bands to 1
};


// Filterbank
q.fcMinERBS = ~f2erbs.(p.fcMin);
q.fcMaxERBS = ~f2erbs.(p.fcMax);
q.nSegments = p.nBands-1;
q.bwERB = (q.fcMaxERBS - q.fcMinERBS) / q.nSegments;
q.fcERBS = q.fcMinERBS + ((0..q.nSegments)*q.bwERB); // Noisternig2017, eq. 3.4
q.fc = ( (0.00437*p.fcMin + 1)*(((0.00437*p.fcMax + 1)/(0.00437*p.fcMin + 1))**((0..q.nSegments)/q.nSegments)) - 1 ) / 0.00437; // Noisternig2017, eq. 3.5
q.fcERB = ~f2erb.(q.fc); // equivalent rectangular bandwidths
q.bw = q.bwERB*q.fcERB; // bandwidths in Hz
// decide if sign should be changed for resynthesis (Noisternig2017)
q.altSigns = (-1 == sign(cos(2*p.fOrder*atan(sqrt((10**(p.fAtt/(10*p.fOrder))) - 1)))));
if ( q.altSigns,
	{q.fSigns = (mod((1..p.nBands),2)*2 - 1)}, // alternating signs
	{q.fSigns = 1!p.nBands} // no sign changes
);
q.stepERBS = q.fcERBS[1]-q.fcERBS[0]; // distance between bands in ERBS


q.normvalues = ~gaussianFunction.((1..p.nBands), q.stepERBS, p.sigma); // pre-computation of weighting values via gaussian function
q.normvalues = q.normvalues.addFirst(0);


q.gammas = abs((0..(p.nBands+1))-(1..p.nBands).flop).collect{|item,i| q.normvalues[item]}; // weights (size B x B+2)

// normalize
q.gammas = q.gammas.flop.collect{ |item,i|
	var gammasL = item[..i];
	var gammasR = item[(i+2)..];
	gammasL = 0.5*gammasL/gammasL.sum;
	gammasR = 0.5*gammasR/gammasR.sum;
	gammasL ++ 0 ++ gammasR;
}.flop;

q.gammas[2] = q.gammas[2] + q.gammas[0]; // combine sub-bands 0 and 2 into 2
q.gammas[p.nBands-1] = q.gammas[p.nBands-1] + q.gammas[p.nBands+1]; // combine sub-bands K+1 and K-1 into K-1
q.gammas = q.gammas[1..p.nBands]; // remove virtual bands 0 and K+1
q.gammas = q.gammas.flop; // q.gammas[i] contains the channel weightings for the i-th channel


// Get Gammatone filter magnitude spectra, after [Hohmann 2002]:
q.beta =  2 * pi * q.fc / s.sampleRate; // Eq. 10
q.phi = pi * q.bw / s.sampleRate; // Eq. 12, line 4
q.u = p.fAtt.neg/p.fOrder; // Eq. 12, line 3
q.p = ((2 * (10**(q.u/10)) * cos(q.phi)) - 2) / (1 - (10**(q.u/10))); // Eq. 12, line 2
q.lambda = (q.p/2).neg - sqrt((q.p.squared/4) - 1); // Eq. 12, line 1
q.fCoeff = Complex(q.lambda, 0) * exp(Complex(0, q.beta)); // Eq. 1, line 2
q.fNorm = 2 * ( (1 - q.fCoeff.abs)**p.fOrder ); // normalization. Sec. 2.2, in text
q.binFreqs = (0..((p.nfft/2))) * s.sampleRate / p.nfft; // bin frequencies in Hz
q.binFreqsMidi = q.binFreqs.cpsmidi; // bin frequencies in midi note numbers (for GUI)
q.z = exp(Complex(0, 2*pi*q.binFreqs/s.sampleRate)); // normalized bin frequencies
q.zCoef = p.nBands.collect({|i| Complex(q.fCoeff.real[i], q.fCoeff.imag[i])/q.z});
q.magResponse = abs( ((1 - q.zCoef)**p.fOrder.neg) * q.fNorm ); // magnitude response at frequency z
q.binBands = q.magResponse.flop.collect{|item,i| item.maxIndex}; // for each bin: take band with maximum amplitude

q.sampleBufs.do{|item,i| item.free}; // free, just in case
q.sampleBufs = 2.collect{Buffer.alloc(s, 65536, 1)}; // allocate a disk i/o buffer
y = Array.newClear(2); // Array of synths for recording

q.sample = [
	~loadWAV.(~path.pathOnly++"wav/sampleA.wav"), // init with buffer A sample
	~loadWAV.(~path.pathOnly++"wav/sampleB.wav") // init with buffer B sample
];

~recomputeDiff.();
~recomputeWeights.();


// test recorder (perhaps move to another file)
q.testRecBuf.free; // free, just in case
q.testRecBuf = Buffer.alloc(s, 65536, 4); // allocate a disk i/o buffer

"done.".postln;