//
// Functions for ACE
//
"ACE: loading functions ... ".post;

~f2erb = { |f| 24.7 * ((0.00437*f) + 1) }; // frequency to ERB
~f2erbs = { |f| 21.4*log10(1 + (0.00437*f)) }; // frequency to ERB-scale (ERBS)
~erbs2f = { |erbs| ((10**(erbs/21.4)) - 1) / 0.00437 }; // ERBS to frequency
~bw2q = { |f, bw| f/bw }; // -3dB bandwidth to Q-factor
~rt2tau = { |t60| t60/log(1000) }; // -60dB reverberation time RT60  to time constant
~tau2rt = { |tau| tau*log(1000) }; // time constant to RT60
~gaussianFunction = { |band,stepERBS,sigma| exp( ((band*stepERBS)**2).neg / (2*(sigma**2)) )}; // Gaussian function
~scp2rho = {|scp| if(scp<=0.5, {26.pow(2*scp.max(0)) - 1}, {25})}; // rho from combined spectral contrast parameter
~scp2beta = {|scp| if(scp<=0.5, {0}, {10.pow(2*(scp.min(1)-0.5)) - 1})}; // beta from cmobined spectral contrast parameter
~rhobeta2scp = {|rho,beta| if(rho<25, {log(rho.max(0)+1)/(2*26.log)}, {log(beta.max(0)+1)/(2*10.log) + 0.5})}; //rho or beta to combined spectral contrast parameter scp

// function to get STFT from 1d signal
~getSTFT = {|x,nfft,winlen| // nfft must be >= winlen! Signal length must be >= winlen!
	var nfft2 = (nfft/2).asInteger; // half fft size
	var winlen2 = (winlen/2).asInteger; // half block size
	var xlen = x.size; // signal length
	var nhalfblocks = (xlen/winlen).floor.asInteger; // number of half blocks
	var nwins = nhalfblocks-1; // number of full blocks that fit in (with 50% overlap)
	var win = Signal.hanningWindow(nfft, nfft-winlen); // hann window
	var costab = Signal.fftCosTable(nfft); // cosine table, needed for fft
	var y = Array.fill(nwins, {Signal.newClear(nfft)}); // array pre-allocation
	var imag = Signal.newClear(nfft); // zero imaginary part
	var normalization = 2*nfft/winlen; // normalization gain factor
	y = nwins.collect{|i| // output spectrum
		var blockStart = i*winlen2; // start index of next block
		var real = win * x[blockStart..(blockStart+nfft-1)]; // windowed signal block
		normalization * fft(real, imag, costab); // normalized spectrum
	};
};

// Load wav file into Signal
~loadWAV = {|path|
	var file, data, signal;
	file = SoundFile.new; // empty sound file
	file.openRead(path); // open path into sound file
	data = FloatArray.newClear(file.numFrames); // init float array
	file.readData(data); // read sound file into float array
	signal = Signal.newClear(file.numFrames); // init signal array
	signal.waveFill({|x,old,i| data[i]}, 0, file.numFrames-1); // output signal
};

// magnitude spectra from STFT
~stft2mags = {|stft|
	stft.collect{|x|
		var mag = x.magnitude;
		mag[0..((mag.size/2).asInteger)];
	};
};

// long-term average magnitude spectrum from signal
~signal2magLT = {|signal,nfft,blockSize|
	var stft = ~getSTFT.(signal, nfft, blockSize) / nfft; // spectra of blocks
	var mags = ~stft2mags.(stft); // magnitude spectra of blocks
	mags.squared.mean.sqrt; // average magnitude spectrum
};


"done.".postln;