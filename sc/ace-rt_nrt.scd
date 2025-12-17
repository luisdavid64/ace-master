"ACE: loading NRT ... ".post;

(
// compile
SynthDef(\ace_nrt, { |bufnum=0,
    dBregDelta= -96, fHPF=50, fHSF=8000, sHSF=0.1, dbHSF=0,
    fHPFtc=4000, tauAtc=5, tauDtc=7, nu= -40, dbNoise= -96,
    tauLI=7, beta=8, mu= -3, rho=30, tauAdp=7,
    t60atCutoff=0.5, dpCutoff=1000.0, tauSP=2, dimWeight=0.5,
    wet=1, vol=0, tauEX=7, lagt60=0.1 |

    var bandWeights = \bandWeights.kr(1.0!p.nBands, lagt60!p.nBands).clip(0, 1);
    var logk = log(1000)/1000, exT60=tauEX*logk, tcT60D=tauDtc*logk, tcT60A=tauAtc*logk;
    var spT60=tauSP*logk, liT60=tauLI*logk, dpT60A=tauAdp*logk;
    var nBands=p.nBands, nBandsLI=p.nBandsLI, fSigns=q.fSigns, fcs=q.fc, bws=q.bw, gammas=q.gammas;
    var dpT60=(fcs.reciprocal*dpCutoff*t60atCutoff).clip(0, t60atCutoff);
    var regDelta=dBregDelta.dbamp, rsHSF=sHSF.reciprocal;

    var sig, sigTC, sigSC, envTCe, ck, env0, env, envS, envSmax, sigACE, out;

    //sig = PlayBuf.ar(2, bufnum, BufRateScale.kr(bufnum), doneAction: 2);
	sig = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum), doneAction:2);
	sig = sig ! 2;
    sig = HPF.ar(sig, fHPF);

    sigTC = HPF.ar(sig, fHPFtc);
    envTCe = Amplitude.ar(sigTC.abs, 0, tcT60D);
    envTCe = (envTCe - Amplitude.ar(envTCe, tcT60A, 0) - nu.dbamp).clip(0, 1);
    sigTC = sigTC * ( envTCe / (Amplitude.ar(envTCe, 0, tcT60D) + regDelta) );

    sigSC = BHiShelf.ar(sig, fHSF, rsHSF, dbHSF);
    sigSC = sigSC + (PinkNoise.ar(16!2).fold(-1,1) * dbamp(dbNoise));
    ck = sigSC.collect{|item,i| CGammatone.ar(item, fcs, bws).flop };

    env0 = 2.collect{|i| ck[i].squared.sum.sqrt.clip(0,1)};
    env  = ck.sum.madd(0.5).squared.sum.sqrt.clip(0,1);

    envS = Amplitude.ar(env, liT60, liT60);
    env = env * ((envS / (nBands.collect{|i|
        var iRange=(0.max(i-nBandsLI)..(nBands-1).min(i+nBandsLI));
        (envS[iRange].squared*gammas[i][iRange].normalizeSum).sum;
    }.sqrt + regDelta))**rho).clip(0,1);

    envS = Amplitude.ar(env, exT60, exT60);
    envSmax = ArrayMax.ar(envS)[0];
    env = env * ((envS/(mu.dbamp*envSmax + regDelta))**beta).clip(0, envSmax/(envS+regDelta));

    envS = Amplitude.ar(env, dpT60A, 0);
    env = (env-envS) + nBands.collect{|i| Amplitude.ar(envS[i], 0, dpT60[i]) };

    env0 = 2.collect{|i| Amplitude.ar(env0[i], spT60, spT60)};
    env  = Amplitude.ar(env, spT60, spT60);

    sigACE = 2.collect{|i| ( ck[i][0] * fSigns * env * bandWeights / (env0[i] + regDelta) ).sum };
    sigACE = BHiShelf.ar(sigACE, fHSF, rsHSF, dbHSF.neg);
    sigACE = XFade2.ar(sigTC, sigACE, (dimWeight*2)-1);
    sigACE = HPF.ar(sigACE, fHPF);

    out = XFade2.ar(sig, sigACE, (wet*2)-1);
    Out.ar(0, (out * vol.dbamp).clip(-1,1));
})
)

(
var dir;
dir = "/Users/luisreyes/Sonify/ace-master/sc/".standardizePath;

// ensure folder exists
if(PathName(dir).isFolder.not) { PathName(dir).mkdir; };

// compile + write synthdef into that folder
SynthDef(\ace_nrt, { |bufnum=0,
    dBregDelta = (-96), fHPF=50, fHSF=8000, sHSF=0.1, dbHSF=0,
    fHPFtc=4000, tauAtc=5, tauDtc=7, nu = (-40), dbNoise = (-96),
    tauLI=7, beta=8, mu = (-3), rho=30, tauAdp=7,
    t60atCutoff=0.5, dpCutoff=1000.0, tauSP=2, dimWeight=0.5,
    wet=1, vol=0, tauEX=7, lagt60=0.1 |

    // ... your ACE body unchanged ...
}).writeDefFile(dir);

("Wrote: " ++ (dir ++ "ace_nrt.scsyndef")).postln;
)

