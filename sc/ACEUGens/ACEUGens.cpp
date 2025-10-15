//SuperCollider is under GNU GPL version 3, https://supercollider.github.io/
//these extensions released under the same license

/*
 *  ACEUGens.cpp
 *  Created by Marian Weger on 2019-07-16
 *
 *  CGammatone is based on Gammatone.cpp, 
 *  created by Nicholas Collins on 1/07/2010.
 *
 */


#include "SC_PlugIn.h"


InterfaceTable *ft; 


//based on V Hohmann Frequency analysis and synthesis using a Gammatone filterbank Acta Acustica vol 88 (2002): 433--442
//converted to straight struct form for SuperCollider from my own GammatoneComplexBandpass class code
struct CGammatone : public Unit  
{
	double centrefrequency; 
	double bandwidth; 
	double normalisation; 
	double reala, imaga; 
	double oldreal[4];
	double oldimag[4]; 
	
	
};



extern "C" {  
	
	void CGammatone_next(CGammatone *unit, int inNumSamples);
	void CGammatone_Ctor(CGammatone* unit);
	//void CGammatone_Dtor(CGammatone* unit);

}



//assumes audio rate, else auditory frequencies make less sense
void CGammatone_Ctor(CGammatone* unit) {
	
	
	for(int i=0; i<4; ++i) {
		unit->oldreal[i]=0.0;  
		unit->oldimag[i]=0.0;  
	}
	
	
	float centrefrequency= ZIN0(1);
	float bandwidth= ZIN0(2);

	
	float samplingrate = unit->mRate->mSampleRate;
	
	double samplingperiod= 1.0/samplingrate;
	
	float nyquist= samplingrate*0.5; 
	
	if (centrefrequency< 20.0) centrefrequency = 20.0; 
	if (centrefrequency>nyquist) centrefrequency = nyquist;  
	
	if ((centrefrequency-(0.5*bandwidth))<1.0) bandwidth = 2.0*(centrefrequency-1.0); 
	
	//if ((centrefrequency+(0.5*bandwidth))>nyquist) bandwidth = 
	
	if (bandwidth>nyquist) bandwidth= nyquist; //assuming there is even room! 
		
	
	
	unit->centrefrequency = centrefrequency; 
	
	//actually need to convert ERBs to 3dB bandwidth
	// bandwidth= 0.887*bandwidth; //converting to 3dB bandwith in Hz, (-3db bandwidth is 0.887 its erb)//[PattersonHoldsworth1996, p.3]
	unit->bandwidth= bandwidth; 
	

	// filter coefficients to calculate, p.435 hohmann paper
	
	double beta= 6.2831853071796*centrefrequency*samplingperiod;
	double phi= 3.1415926535898*bandwidth*samplingperiod;
	// double u= pow(10, 4*0.25*(-0.1)); // order hardcoded to 4, attenuation to 4dB
	// double p=  (1.6827902832904*cos(phi) -2)*6.3049771007832; // for 3dB attenuation and order 4
	// double p= (2*u*cos(phi) - 2) / (1-u);
	double p= (1.5886564694486*cos(phi) - 2) * 4.8621160938616; // for 4dB attenuation and order 4
	double lambda= (p*(-0.5)) - sqrt(p*p*0.25 - 1.0); 
	
	unit->reala= lambda*cos(beta); 
	unit->imaga= lambda*sin(beta);
	
	//avoid b= 0 or Nyquist, otherise must remove factor of 2.0 here
	unit->normalisation= 2.0*(pow(1-fabs(lambda),4)); 	
	
	
	//printf("set-up gammatone filter %f %f %f %f %f \n",centrefrequency, bandwidth, unit->normalisation, unit->reala, unit->imaga);
	
	
	SETCALC(CGammatone_next);
	
}


// this is a function for preventing pathological math operations in ugens.
// can be used at the end of a block to fix any recirculating filter values.
//inline float zapgremlins(float x)
//{
//	float absx = fabs(x);
//	// very small numbers fail the first test, eliminating denormalized numbers
//	//    (zero also fails the first test, but that is OK since it returns zero.)
//	// very large numbers fail the second test, eliminating infinities
//	// Not-a-Numbers fail both tests and are eliminated.
//	return (absx > (float)1e-15 && absx < (float)1e15) ? x : (float)0.;
//}

void CGammatone_next(CGammatone *unit, int inNumSamples) {
	
	int i,j; 
	
	float *input = IN(0); 
	float *outputreal = OUT(0);
	float *outputimag = OUT(1);
	
	double newreal, newimag; 
	
	double * oldreal = &(unit->oldreal[0]); 
	double * oldimag = &(unit->oldimag[0]); 
	double reala = unit->reala; 
	double imaga = unit->imaga; 
	double normalisation = unit->normalisation; 
	
	for (i=0; i<inNumSamples; ++i) {
		
		newreal= input[i]; //real input 
		newimag=0.0; 
		
		for (j=0; j<4; ++j) {
			
			newreal= newreal + (reala*oldreal[j])-(imaga*oldimag[j]);
			newimag= newimag + (reala*oldimag[j])+(imaga*oldreal[j]);
			
			oldreal[j]= newreal; //zapgremlins(newreal); //trying to avoid denormals which mess up processing via underflow
			oldimag[j]= newimag; //zapgremlins(newimag); 
		}
		
		outputreal[i]= newreal*normalisation; 
		
		//imaginary output too could be useful

		outputimag[i]= newimag*normalisation; 
		
	}
	
	//printf("testing a %f %f %f %f \n",newreal, newimag, normalisation, output[0]);
	
}

//void CGammatone_Dtor(CGammatone* unit) {
//	
//	
//}



PluginLoad(ACEUGens) {
	
	ft = inTable;
	
	//DefineSimpleCantAliasUnit(PitchNoteUGen);

	DefineSimpleUnit(CGammatone);
	
	
	//see http://www.listarc.bham.ac.uk/lists/sc-dev-2003/msg03275.html
	//DefinePlugInCmd; at scope of whole plugin
	
	//DefineUnitCmd; at scope of single UGen; I believe which node is passed in so can be specific to one running instance 
	
}





