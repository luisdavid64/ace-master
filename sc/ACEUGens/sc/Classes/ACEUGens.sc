
CGammatone : MultiOutUGen {	
	*ar { 
		arg input, centrefrequency=440.0, bandwidth=200.0, mul=1.0, add=0.0;
		^this.multiNew('audio', input, centrefrequency, bandwidth).madd(mul, add)
	}
	init { arg ... theInputs;
		inputs = theInputs;
		channels = [
			OutputProxy(rate, this, 0),
			OutputProxy(rate, this, 1)
		];
		^channels
	}
	checkInputs { ^this.checkNInputs(1) }
}