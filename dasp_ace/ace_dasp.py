"""
Optimized ACE Implementation using DASP PyTorch - Version 2
Focus on using actual DASP components and efficient processing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import dasp_pytorch.signal_dasp as dasp_signal
import dasp_pytorch.functional as dasp_F
# from high_pass import DifferentiableHPF  # Commented out - not needed
import torchaudio


class OptimizedDASPACE(nn.Module):
    """
    Complete ACE implementation with all SuperCollider parameters
    Based on the original ACE SynthDef arguments
    """
    
    def __init__(self, 
                 sample_rate: float = 44100,
                 n_bands: int = 32,  # Fewer bands for efficiency
                 filter_duration: float = 0.02):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        
        # Use DASP gammatone filterbank
        raw_filterbank = dasp_signal.gammatone_filterbank(
            sample_rate=int(sample_rate),
            num_bands=n_bands,
            low_freq=50.0,
            high_freq=sample_rate // 2,
            order=4,
            duration=filter_duration
        )
        
        # CRITICAL FIX: Prevent massive over-amplification with simple scaling
        # The raw DASP filters cause ~100x amplification. Apply a conservative
        # scaling factor to bring it to reasonable levels.
        conservative_scale = 0.16  # Targeting ~1.2x enhancement
        self.filterbank = raw_filterbank * conservative_scale
        
        print(f"Filterbank scaling: applied {conservative_scale}x factor to prevent over-amplification")
        
        # All SuperCollider ACE parameters as learnable parameters
        # Core filter parameters
        self.dBregDelta = nn.Parameter(torch.tensor(-96.0))     # regularization delta in dB
        self.fHPF = nn.Parameter(torch.tensor(50.0))            # high-pass filter frequency
        self.fHSF = nn.Parameter(torch.tensor(8000.0))          # high-shelf filter frequency
        self.sHSF = nn.Parameter(torch.tensor(0.1))             # high-shelf filter slope
        self.dbHSF = nn.Parameter(torch.tensor(0.0))            # high-shelf filter gain in dB
        self.fHPFtc = nn.Parameter(torch.tensor(4000.0))        # TCE high-pass filter frequency
        
        # Temporal Contrast Enhancement (TCE) parameters
        self.tauAtc = nn.Parameter(torch.tensor(3.0))  # ms, enva
        self.tauDtc = nn.Parameter(torch.tensor(7.0))  # ms, envd
        self.nu      = nn.Parameter(torch.tensor(-40.0))  # dB relative threshold (good starting point)
        
        # Noise parameters
        self.dbNoise = nn.Parameter(torch.tensor(-96.0))        # noise level in dB
        
        # Leaky Integrator (LI) parameters
        self.tauLI = nn.Parameter(torch.tensor(7.0))            # LI time constant
        
        # Enhancement (EX) parameters
        self.beta = nn.Parameter(torch.tensor(8.0))             # EX exponent
        self.mu = nn.Parameter(torch.tensor(-3.0))              # EX scaling factor in dB
        self.rho = nn.Parameter(torch.tensor(30.0))             # EX power law exponent
        self.tauEX = nn.Parameter(torch.tensor(7.0))            # EX time constant
        
        # Dynamic Programming (DP) parameters
        self.tauAdp = nn.Parameter(torch.tensor(7.0))           # DP adaptation time constant
        self.t60atCutoff = nn.Parameter(torch.tensor(0.5))      # T60 at cutoff frequency
        self.dpCutoff = nn.Parameter(torch.tensor(1000.0))      # DP cutoff frequency
        
        # Summation and mixing parameters
        self.tauSP = nn.Parameter(torch.tensor(2.0))            # summation phase time constant
        self.dimWeight = nn.Parameter(torch.tensor(0.5))        # spectral vs temporal weight
        self.wet = nn.Parameter(torch.tensor(1.0))              # dry-wet mix
        self.vol = nn.Parameter(torch.tensor(0.0))              # master volume in dB
        self.lagt60 = nn.Parameter(torch.tensor(0.1))           # lag time for T60
        
        # Band-specific gains (learnable per frequency band)
        self.band_gains = nn.Parameter(torch.ones(n_bands))     # band weights
        # self.hpf_module = DifferentiableHPF(sample_rate)  # Commented out


        
    """ Temporal Contrast Enhancement (TCE) Functions """

    def _time_constant_to_alpha_T60(self, tau_ms: torch.Tensor) -> torch.Tensor:
        """
        Match SuperCollider's mapping:
        logk = log(1000)/1000
        T60  = tau_ms * logk  (seconds)
        alpha = exp(-ln(1000) / (T60 * fs))

        Result: same effective behavior as Amplitude.ar with T60 = tau_ms*logk.
        """
        fs = torch.tensor(float(self.sample_rate), device=tau_ms.device, dtype=tau_ms.dtype)
        ln1000 = torch.log(torch.tensor(1000.0, device=tau_ms.device, dtype=tau_ms.dtype))
        logk = ln1000 / 1000.0

        tau_ms = torch.clamp(tau_ms, 0.1, 1000.0)
        T60 = tau_ms * logk / 1000.0   # seconds

        alpha = torch.exp(-ln1000 / (T60 * fs))
        return alpha

 
    def _nonlinear_envelope(self, x: torch.Tensor, alpha: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Non-linear envelope follower as in Eqs. (9–10).

        x:    [batch, channels, time]
        alpha: scalar tensor (0D) smoothing factor
        mode: "attack" -> enva  (smooth attack, instant decay)
              "decay"  -> envd  (smooth decay, instant attack)
        """
        assert x.dim() == 3, "Expected [batch, channels, time]"
        B, C, T = x.shape
        x_abs = x.abs()

        # Ensure alpha is on the right device/dtype
        alpha = alpha.to(dtype=x.dtype, device=x.device)
        one_minus_alpha = 1.0 - alpha

        # Output and previous state
        y = torch.empty_like(x_abs)
        y_prev = torch.zeros(B, C, device=x.device, dtype=x.dtype)

        for n in range(T):
            xn = x_abs[..., n]  # [B, C]

            if mode == "attack":
                # enva: smooth ATTACK (rising), instant DECAY (falling)
                smooth_cond = xn > y_prev
            elif mode == "decay":
                # envd: smooth DECAY (falling), instant ATTACK (rising)
                smooth_cond = xn < y_prev
            else:
                raise ValueError(f"Unknown mode {mode}")

            # Smooth where condition holds, otherwise follow input instantly
            y_smooth = one_minus_alpha * xn + alpha * y_prev
            y_curr = torch.where(smooth_cond, y_smooth, xn)

            y[..., n] = y_curr
            y_prev = y_curr

        return y

    def enva(self, x: torch.Tensor, tau_ms: torch.Tensor) -> torch.Tensor:
        """Envelope with smooth attack, instant decay."""
        alpha = self._time_constant_to_alpha_T60(tau_ms)
        return self._nonlinear_envelope(x, alpha, mode="attack")

    def envd(self, x: torch.Tensor, tau_ms: torch.Tensor) -> torch.Tensor:
        """Envelope with smooth decay, instant attack."""
        alpha = self._time_constant_to_alpha_T60(tau_ms)
        return self._nonlinear_envelope(x, alpha, mode="decay")

    
    def high_pass_filter(self, x: torch.Tensor, freq: torch.Tensor, sample_rate: float) -> torch.Tensor:
        """Simple high-pass filter implementation for 3D tensors [batch, channels, time]"""
        B, C, T = x.shape
        # Process each batch item separately since torchaudio expects 2D
        output = torch.empty_like(x)
        for b in range(B):
            # torchaudio expects [channels, time]
            output[b] = torchaudio.functional.highpass_biquad(x[b], sample_rate, freq)
        return output

    def _design_high_shelf(self,
                        freq: torch.Tensor,
                        slope: torch.Tensor,
                        gain_db: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        RBJ high-shelf filter design.
        Returns normalized biquad coefficients (b0,b1,b2,a1,a2), a0 assumed = 1.
        All scalars as 0D tensors on correct device/dtype.
        """
        fs  = torch.tensor(float(self.sample_rate),
                        device=freq.device, dtype=freq.dtype)
        # clamp
        f0  = freq.clamp(20.0, fs/2 - 100.0)
        S   = slope.clamp(0.1, 2.0)
        A   = torch.pow(10.0, gain_db / 40.0)      # 10^(dBgain/40)
        w0  = 2.0 * math.pi * f0 / fs
        cosw0 = torch.cos(w0)
        sinw0 = torch.sin(w0)

        # RBJ alpha
        tmp = (A + 1.0 / A) * (1.0 / S - 1.0) + 2.0
        alpha = sinw0 / 2.0 * torch.sqrt(tmp)

        sqrtA = torch.sqrt(A)

        b0 =    A * ((A + 1.0) + (A - 1.0) * cosw0 + 2.0 * sqrtA * alpha)
        b1 = -2*A * ((A - 1.0) + (A + 1.0) * cosw0)
        b2 =    A * ((A + 1.0) + (A - 1.0) * cosw0 - 2.0 * sqrtA * alpha)
        a0 =        (A + 1.0) - (A - 1.0) * cosw0 + 2.0 * sqrtA * alpha
        a1 =  2.0 * ((A - 1.0) - (A + 1.0) * cosw0)
        a2 =        (A + 1.0) - (A - 1.0) * cosw0 - 2.0 * sqrtA * alpha

        # normalize so a0 = 1
        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0

        return b0, b1, b2, a1, a2
    
    def high_shelf_filter(self,
                        x: torch.Tensor,
                        freq: torch.Tensor,
                        slope: torch.Tensor,
                        gain_db: torch.Tensor) -> torch.Tensor:
        """
        Differentiable biquad high-shelf filter.
        x: [B, C, T]
        freq, slope, gain_db: scalar tensors (same across batch/channels)
        """
        B, C, T = x.shape
        device, dtype = x.device, x.dtype

        # design biquad coeffs (scalars)
        b0, b1, b2, a1, a2 = self._design_high_shelf(
            freq.to(device=device, dtype=dtype),
            slope.to(device=device, dtype=dtype),
            gain_db.to(device=device, dtype=dtype)
        )

        # flatten batch+channels -> [N, T]
        N = B * C
        x_flat = x.reshape(N, T)

        # DF-II Transposed
        y_flat = torch.zeros_like(x_flat)
        z1 = torch.zeros(N, device=device, dtype=dtype)
        z2 = torch.zeros(N, device=device, dtype=dtype)

        for n in range(T):
            xn = x_flat[:, n]
            yn = b0 * xn + z1
            z1_new = b1 * xn - a1 * yn + z2
            z2     = b2 * xn - a2 * yn
            z1     = z1_new
            y_flat[:, n] = yn

        y = y_flat.view(B, C, T)
        return y

    
    def db_to_amplitude(self, db: torch.Tensor) -> torch.Tensor:
        """Convert dB to linear amplitude"""
        return torch.pow(10.0, db / 20.0)

    def amplitude_env(self, x: torch.Tensor, attack_tau_ms: torch.Tensor, release_tau_ms: torch.Tensor) -> torch.Tensor:
        """
        Emulate SuperCollider Amplitude.ar with T60-mapped times.

        x: [B, C, T] or [B, C, K, T]
        """
        if x.dim() == 3:
            B, C, T = x.shape
            x_abs = x.abs()

            alphaA = self._time_constant_to_alpha_T60(attack_tau_ms).to(x.device, x.dtype)
            alphaR = self._time_constant_to_alpha_T60(release_tau_ms).to(x.device, x.dtype)

            y = torch.empty_like(x_abs)
            y_prev = torch.zeros(B, C, device=x.device, dtype=x.dtype)

            for n in range(T):
                xn = x_abs[..., n]
                # Different coefficients for rising vs falling
                rising = xn > y_prev
                alpha = torch.where(rising, alphaA, alphaR)
                one_minus = 1.0 - alpha
                y_curr = one_minus * xn + alpha * y_prev
                y[..., n] = y_curr
                y_prev = y_curr
            return y

        elif x.dim() == 4:
            B, C, K, T = x.shape
            x_abs = x.abs()

            alphaA = self._time_constant_to_alpha_T60(attack_tau_ms).to(x.device, x.dtype)
            alphaR = self._time_constant_to_alpha_T60(release_tau_ms).to(x.device, x.dtype)

            y = torch.empty_like(x_abs)
            y_prev = torch.zeros(B, C, K, device=x.device, dtype=x.dtype)

            for n in range(T):
                xn = x_abs[..., n]
                rising = xn > y_prev
                alpha = torch.where(rising, alphaA, alphaR)
                one_minus = 1.0 - alpha
                y_curr = one_minus * xn + alpha * y_prev
                y[..., n] = y_curr
                y_prev = y_curr
            return y

        else:
            raise ValueError("amplitude_env expects 3D or 4D input")

    
    def temporal_contrast_enhancement(self, x: torch.Tensor) -> torch.Tensor:
        """
        Temporal ACE exactly following the SuperCollider SynthDef:
        - sigTC = HPF(sig, fHPFtc)
        - envTCe = Amplitude(sigTC.abs, 0, tcT60D)
        - envTCe = (envTCe - Amplitude(envTCe, tcT60A, 0) - nu.dbamp).clip(0, 1)
        - sigTC = sigTC * (envTCe / (Amplitude(envTCe, 0, tcT60D) + regDelta))
        """
        # sigTC = HPF(sig, fHPFtc)
        tce_hpf_freq = torch.clamp(self.fHPFtc, 100.0, self.sample_rate / 2.0)
        sigTC = self.high_pass_filter(x, tce_hpf_freq, self.sample_rate)

        # Convert tau values to T60 using SuperCollider's logk factor
        logk = torch.log(torch.tensor(1000.0, device=x.device, dtype=x.dtype)) / 1000.0
        tcT60D = torch.clamp(self.tauDtc, 1.0, 100.0) * logk * 1000.0  # Convert to ms
        tcT60A = torch.clamp(self.tauAtc, 1.0, 100.0) * logk * 1000.0  # Convert to ms

        # envTCe = Amplitude(sigTC.abs, 0, tcT60D)
        envTCe = self.amplitude_env(sigTC.abs(), torch.tensor(0.0, device=x.device, dtype=x.dtype), tcT60D)

        # envTCe = (envTCe - Amplitude(envTCe, tcT60A, 0) - nu.dbamp).clip(0, 1)
        # Note: envTCe is reused as both input and output!
        envTCe_attack = self.amplitude_env(envTCe, tcT60A, torch.tensor(0.0, device=x.device, dtype=x.dtype))
        nu_amp = self.db_to_amplitude(self.nu.clamp(-60.0, 0.0)).to(x.device, x.dtype).view(1, 1, 1)
        envTCe = (envTCe - envTCe_attack - nu_amp).clamp(0.0, 1.0)

        # sigTC = sigTC * (envTCe / (Amplitude(envTCe, 0, tcT60D) + regDelta))
        envTCe_denominator = self.amplitude_env(envTCe, torch.tensor(0.0, device=x.device, dtype=x.dtype), tcT60D)
        regDelta = self.db_to_amplitude(self.dBregDelta.clamp(-120.0, -60.0)).to(x.device, x.dtype).view(1, 1, 1)

        sigTC_enh = sigTC * (envTCe / (envTCe_denominator + regDelta))

        return sigTC_enh

 
    
    """ Spectral Contrast Enhancement (SCE) Functions """
    
        # ===== ERB utilities =====
    def _hz_to_erb(self, f_hz: torch.Tensor) -> torch.Tensor:
        """
        Glasberg & Moore ERB-rate approximation.
        """
        return 21.4 * torch.log10(4.37e-3 * f_hz + 1.0)

    def _erb_space(self,
                   f_min: float,
                   f_max: float,
                   n_bands: int,
                   device,
                   dtype) -> torch.Tensor:
        """
        Center frequencies equally spaced on ERB-rate scale.
        """
        f_min = torch.tensor(f_min, device=device, dtype=dtype)
        f_max = torch.tensor(f_max, device=device, dtype=dtype)

        erb_min = self._hz_to_erb(f_min)
        erb_max = self._hz_to_erb(f_max)

        erb = torch.linspace(erb_min, erb_max, n_bands,
                             device=device, dtype=dtype)
        # inverse ERB-rate
        f = (torch.pow(10.0, erb / 21.4) - 1.0) / 4.37e-3
        return f

    # ===== simple linear leaky integrator (for LP blocks) =====
    def leaky_integrator(self, x: torch.Tensor,
                         tau_ms: torch.Tensor) -> torch.Tensor:
        """
        First-order leaky integrator along time:
            y[n] = (1 - alpha) x[n] + alpha y[n-1]
        tau_ms: scalar (ms)
        """
        assert x.dim() in (3, 4), "Expect [B,C,T] or [B,C,K,T]"
        alpha = self._time_constant_to_alpha(tau_ms)
        alpha = alpha.to(device=x.device, dtype=x.dtype)
        one_minus_alpha = 1.0 - alpha

        if x.dim() == 3:
            B, C, T = x.shape
            y = torch.empty_like(x)
            y_prev = torch.zeros(B, C, device=x.device, dtype=x.dtype)
            for n in range(T):
                xn = x[..., n]
                y_curr = one_minus_alpha * xn + alpha * y_prev
                y[..., n] = y_curr
                y_prev = y_curr
            return y

        else:  # [B, C, K, T]
            B, C, K, T = x.shape
            y = torch.empty_like(x)
            y_prev = torch.zeros(B, C, K, device=x.device, dtype=x.dtype)
            for n in range(T):
                xn = x[..., n]
                y_curr = one_minus_alpha * xn + alpha * y_prev
                y[..., n] = y_curr
                y_prev = y_curr
            return y

    def _lateral_inhibition(self, e_tilde: torch.Tensor) -> torch.Tensor:
        """
        Lateral inhibition according to Eqs. (3–6), simplified:
        - Gaussian weights on ERB-rate axis
        - symmetric normalization over all neighbors

        e_tilde: [B, C, K, T] (smoothed envelopes)
        returns T_k[n]: [B, C, K, T]
        """
        assert e_tilde.dim() == 4
        B, C, K, T = e_tilde.shape
        device, dtype = e_tilde.device, e_tilde.dtype

        # Center freqs on ERB scale
        fc = self._erb_space(50.0, self.sample_rate / 2.0,
                             K, device, dtype)              # [K]
        fc_erb = self._hz_to_erb(fc)                        # [K]

        # Gaussian weights in ERB domain
        sigma_erb = getattr(self, "sigmaERB", 3.0)          # ≈ 3 ERB as in paper
        delta = fc_erb.view(K, 1) - fc_erb.view(1, K)       # [K,K]
        weights = torch.exp(-(delta ** 2) /
                            (2.0 * (sigma_erb ** 2)))       # [K,K]

        # No self-inhibition
        weights = weights * (1.0 - torch.eye(K, device=device, dtype=dtype))

        # Normalize each row (sum_i w_{k,i} = 1)
        row_sums = weights.sum(dim=1, keepdim=True) + 1e-12
        weights = weights / row_sums                        # [K,K]

        # T_k^2[n] = sum_i w_{k,i} * e_i^2[n]
        e2 = e_tilde ** 2                                   # [B,C,K,T]
        BC = B * C
        e2_bc = e2.view(BC, K, T)                           # [BC,K,T]

        # weights[k,i] * e2_bc[:,i,t] -> [BC,K,T]
        T2_bc = torch.einsum('ki,b i t -> b k t', weights, e2_bc)
        T2 = T2_bc.view(B, C, K, T)
        T = torch.sqrt(T2 + 1e-12)

        return T

    def _decay_prolongation(self,
                            v: torch.Tensor,
                            band_centers_hz: torch.Tensor) -> torch.Tensor:
        """
        Decay prolongation on v_k[n] according to Eq. (11):

            pk = envd( enva(vk) ) + vk - enva(vk)

        with per-band T60(f_c).
        v: [B, C, K, T]
        band_centers_hz: [K]
        """
        assert v.dim() == 4
        B, C, K, T = v.shape
        device, dtype = v.device, v.dtype

        # T60 at cutoff frequency (seconds) and cutoff freq
        T60_crit = torch.clamp(self.t60atCutoff, 0.01, 5.0).to(device=device, dtype=dtype)
        f_clip = torch.clamp(self.dpCutoff, 100.0, self.sample_rate / 2.0
                             ).to(device=device, dtype=dtype)

        fc = band_centers_hz.to(device=device, dtype=dtype).view(1, 1, K, 1)

        # T60(fc): constant below f_clip, inverse above (paper description)
        T60 = torch.where(fc <= f_clip,
                          T60_crit,
                          T60_crit * (f_clip / fc))        # [1,1,K,1]

        # Convert T60 to alpha via Eq. (10): α = exp(-ln(1000)/(T60 fs))
        fs = torch.tensor(float(self.sample_rate), device=device, dtype=dtype)
        ln1000 = torch.log(torch.tensor(1000.0, device=device, dtype=dtype))
        alpha = torch.exp(-ln1000 / (T60 * fs))             # [1,1,K,1]

        # Use the general non-linear envelope with band-wise alpha
        # Flatten B and C so bands become 'channels'
        x_bc = v.view(B * C, K, T)                          # [BC,K,T]
        alpha_band = alpha.view(K)                          # [K]

        env_a_bc = self._nonlinear_envelope(x_bc, alpha_band, mode="attack")
        env_a = env_a_bc.view(B, C, K, T)

        resid = v - env_a

        env_d_bc = self._nonlinear_envelope(env_a_bc, alpha_band, mode="decay")
        env_d = env_d_bc.view(B, C, K, T)

        p = env_d + resid
        return p

    
    def apply_gammatone_filterbank(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efficient gammatone filtering using DASP
        """
        batch_size, n_channels, n_samples = x.shape
        
        # Move filterbank to device
        filters = self.filterbank.to(x.device)
        
        # Apply filtering efficiently
        outputs = []
        for ch in range(n_channels):
            channel_data = x[:, ch:ch+1, :]  # [batch, 1, time]
            
            # Apply all filters at once using grouped convolution
            # Reshape filters for grouped conv: [n_bands, 1, filter_len]
            filtered = F.conv1d(
                channel_data.repeat(1, self.n_bands, 1),  # [batch, n_bands, time]
                filters.view(self.n_bands, 1, -1),  # [n_bands, 1, filter_len]  
                groups=self.n_bands,
                padding='same'
            )  # [batch, n_bands, time]
            
            outputs.append(filtered)
        
        # Stack channels: [batch, channels, bands, time]
        return torch.stack(outputs, dim=1)
    
    def envelope_follower_with_times(self, x: torch.Tensor, attack_time: float = 0.01, decay_time: float = 0.1) -> torch.Tensor:
        """
        Envelope following with separate attack/decay times following SuperCollider Amplitude.ar
        """
        # Simplified envelope following using averaging (more efficient than manual loops)
        kernel_size = max(3, min(21, int(min(attack_time, decay_time) * self.sample_rate / 10)))
        
        if x.dim() == 4:  # [batch, channels, bands, time]
            # Use reflection padding to maintain size
            padded = F.pad(x.abs(), (kernel_size//2, kernel_size//2), mode='reflect')
            envelope = F.avg_pool2d(padded, kernel_size=(1, kernel_size), stride=1)
            # Ensure same size as input
            if envelope.shape[-1] != x.shape[-1]:
                envelope = envelope[..., :x.shape[-1]]
        else:  # [batch, channels, time]
            # Use reflection padding to maintain size
            padded = F.pad(x.abs(), (kernel_size//2, kernel_size//2), mode='reflect')
            envelope = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
            # Ensure same size as input
            if envelope.shape[-1] != x.shape[-1]:
                envelope = envelope[..., :x.shape[-1]]
        
        return envelope

    def spectral_contrast_enhancement(self, band_signals: torch.Tensor) -> torch.Tensor:
        """
        Spectral ACE (LI + EX + DP) following Marian Weger's SuperCollider SynthDef.

        Args:
            band_signals: [B, C, K, T]  real subband signals from gammatone filterbank
                B = batch, C = channels, K = bands, T = time

        Returns:
            processed subbands c'_k[n]  [B, C, K, T]
        """
        B, C, K, T = band_signals.shape
        device, dtype = band_signals.device, band_signals.dtype
        x = band_signals

        # --- 0. envelopes per channel and mono mix -------------------------------
        env0 = x.abs()                     # [B,C,K,T]
        env  = env0.mean(dim=1, keepdim=True)  # [B,1,K,T]  mono envelope for processing

        # --- 1. Leaky integrator (LI) -------------------------------------------
        tau_li_ms = torch.clamp(self.tauLI, 1.0, 50.0)
        env_s = self.amplitude_env(env, tau_li_ms, tau_li_ms)  # Amplitude.ar(env, liT60, liT60)

        # ERB–Gaussian lateral inhibition
        fc = self._erb_space(50.0, self.sample_rate / 2.0, K, device, dtype)
        fc_erb = self._hz_to_erb(fc)
        sigma_erb = 3.0
        delta = fc_erb.view(K, 1) - fc_erb.view(1, K)
        weights = torch.exp(-(delta ** 2) / (2.0 * sigma_erb ** 2))
        weights = weights * (1.0 - torch.eye(K, device=device, dtype=dtype))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        e2 = env_s.pow(2).squeeze(1)                           # [B,K,T]
        T2 = torch.einsum('ki,b i t -> b k t', weights, e2)    # inhibition term
        T = torch.sqrt(T2 + 1e-12).unsqueeze(1)                # [B,1,K,T]

        rho = torch.clamp(self.rho, 0.0, 100.0)
        regDelta = self.db_to_amplitude(self.dBregDelta.clamp(-120.0, -60.0)).to(device)
        env_li = env * ((env_s / (T + regDelta)) ** rho).clamp(0.0, 1.0)

        # --- 2. Dynamics expansion (EX) -----------------------------------------
        tau_ex_ms = torch.clamp(self.tauEX, 1.0, 100.0)
        env_ex_s = self.amplitude_env(env_li, tau_ex_ms, tau_ex_ms)
        env_ex_smax, _ = env_ex_s.max(dim=2, keepdim=True)  # [B,1,1,T]

        beta = torch.clamp(self.beta, 0.0, 20.0)
        mu_lin = self.db_to_amplitude(self.mu).to(device)
        thresh = mu_lin * env_ex_smax

        gain1 = (env_ex_s / (thresh + regDelta)).clamp(0.0, 10.0) ** beta
        gain2 = (env_ex_smax / (env_ex_s + regDelta)).clamp(0.0, 10.0)
        env_ex = env_li * torch.minimum(gain1, gain2)

        # --- 3. Decay prolongation (DP) -----------------------------------------
        # per-band T60(fc)
        fc = fc.view(1, 1, K, 1)
        T60cut = torch.clamp(self.t60atCutoff, 0.01, 5.0).to(device)
        fcut   = torch.clamp(self.dpCutoff, 100.0, self.sample_rate / 2.0).to(device)
        T60 = torch.where(fc <= fcut, T60cut, T60cut * (fcut / fc))
        fs = torch.tensor(float(self.sample_rate), device=device, dtype=dtype)
        ln1000 = torch.log(torch.tensor(1000.0, device=device, dtype=dtype))
        alpha = torch.exp(-ln1000 / (T60 * fs))               # [1,1,K,1]

        # Apply attack/decay envelope pair with residuum (Eq. 11)
        v = env_ex
        env_a = self.amplitude_env(v, self.tauAdp, torch.tensor(0.0, device=device, dtype=dtype))
        env_d = self.amplitude_env(env_a, torch.tensor(0.0, device=device, dtype=dtype), self.tauAdp)
        env_dp = env_d + (v - env_a)

        # --- 4. Short smoothing before applying ratio (SP) -----------------------
        tau_sp_ms = torch.clamp(self.tauSP, 0.5, 20.0)
        env0_mono = env0.mean(dim=1, keepdim=True)            # [B,1,K,T]
        env_lp0 = self.amplitude_env(env0_mono, tau_sp_ms, tau_sp_ms)
        env_lp  = self.amplitude_env(env_dp,     tau_sp_ms, tau_sp_ms)

        # --- 5. Apply processed envelope back to subbands ------------------------
        env_ratio = env_lp / (env_lp0 + regDelta)
        band_weights = torch.clamp(self.band_gains, 0.0, 5.0).view(1, 1, K, 1)

        c_out = x * env_ratio * band_weights
        return c_out
 
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete ACE processing pipeline faithfully following Marian Weger's SuperCollider implementation.
        """
        # Store original input for dry/wet mix (before ANY processing)
        dry = x.clone()
        
        # 1. Input high-pass (only for wet processing path)
        hpf_freq = torch.clamp(self.fHPF, 20.0, 1000.0)
        sig = self.high_pass_filter(x, hpf_freq, self.sample_rate)

        # ---------------- TCE BRANCH ----------------
        tce_output = self.temporal_contrast_enhancement(sig)

        # ---------------- SCE BRANCH ----------------
        # High-shelf filter (pre-emphasis)
        hsf_freq  = torch.clamp(self.fHSF, 1000.0, self.sample_rate / 2.0)
        hsf_slope = torch.clamp(self.sHSF, 0.1, 2.0)
        hsf_gain  = self.dbHSF
        sigSC = self.high_shelf_filter(sig, hsf_freq, hsf_slope, hsf_gain)

        # Add pink-ish noise (for envelope stability)
        noise_amp = self.db_to_amplitude(self.dbNoise.clamp(-120.0, -60.0))
        sigSC_noisy = sigSC + noise_amp * torch.randn_like(sigSC)

        # Apply gammatone filterbank
        band_signals = self.apply_gammatone_filterbank(sigSC_noisy)  # [B, C, K, T]

        # Apply SCE (LI + EX + DP + SP)
        sce_bands = self.spectral_contrast_enhancement(band_signals)  # [B, C, K, T]

        # Reconstruct broadband signal
        sce_output = torch.sum(sce_bands, dim=2)  # [B, C, T]

        # Post high-shelf compensation (negative gain)
        hsf_gain_neg = -torch.clamp(self.dbHSF, -20.0, 20.0)
        sce_filtered = self.high_shelf_filter(sce_output, hsf_freq, hsf_slope, hsf_gain_neg)

        # Crossfade TCE and SCE (dimWeight)
        dim_weight = torch.clamp(self.dimWeight, 0.0, 1.0)
        enhanced = (1.0 - dim_weight) * tce_output + dim_weight * sce_filtered

        # Final high-pass filter (applied to ACE signal only, before dry/wet mix)
        enhanced = self.high_pass_filter(enhanced, hpf_freq, self.sample_rate)

        # Dry/wet mix (sig = dry signal, enhanced = wet ACE signal)
        wet_amount = torch.clamp(self.wet, 0.0, 1.0)
        output = (1.0 - wet_amount) * dry + wet_amount * enhanced
        
        # Volume (applied once to final mix)
        vol_gain = self.db_to_amplitude(torch.clamp(self.vol, -60.0, 20.0))
        output = output * vol_gain

        # Prevent clipping
        output = torch.clamp(output, -1.0, 1.0)

        # Return all useful intermediate signals for inspection
        return {
            "output": output,          # final dry+wet output
            "dry": dry,                # original
            "wet": enhanced,           # processed ACE signal
            "enhanced": enhanced,      # same as wet for compatibility
            "tce_output": tce_output,  # transient-only
            "sce_output": sce_output,  # reconstructed spectral
            "band_signals": band_signals,  # gammatone subbands
            "sce_bands": sce_bands     # processed subbands
        }
    
    def get_parameters_dict(self) -> Dict[str, float]:
        """Get all ACE parameters as dictionary matching SuperCollider naming"""
        params = {}
        for name, param in self.named_parameters():
            if param.numel() == 1:  # Scalar parameters
                params[name] = param.item()
            else:  # Vector parameters (like band_gains)
                params[f"{name}_mean"] = param.mean().item()
                params[f"{name}_std"] = param.std().item()
        return params


def create_dasp_ace_processor(sample_rate: float = 44100, **kwargs) -> OptimizedDASPACE:
    """
    Factory function to create DASP ACE processor
    """
    return OptimizedDASPACE(sample_rate=sample_rate, **kwargs)


def process_with_optimized_ace(audio: torch.Tensor, 
                              sample_rate: float = 44100,
                              ace_params: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    """
    Process audio with optimized DASP ACE
    """
    processor = create_dasp_ace_processor(sample_rate=sample_rate)
    
    # Set parameters if provided
    if ace_params:
        with torch.no_grad():
            for name, value in ace_params.items():
                if hasattr(processor, name):
                    getattr(processor, name).data.fill_(value)
    
    # Process
    processor.eval()
    with torch.no_grad():
        return processor(audio)


# Test the optimized implementation
if __name__ == "__main__":
    print("Testing Optimized DASP ACE...")
    
    # Create test audio
    sample_rate = 44100
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Stereo chirp signal
    freq = 200 + 1000 * t
    left = torch.sin(2 * np.pi * freq * t)
    right = torch.cos(2 * np.pi * freq * t)
    test_audio = torch.stack([left, right]).unsqueeze(0)
    
    print(f"Input: {test_audio.shape}")
    
    try:
        # Process with optimized ACE
        results = process_with_optimized_ace(test_audio, sample_rate)
        
        print(f"✓ Processing successful!")
        print(f"Output: {results['output'].shape}")
        print(f"Available outputs: {list(results.keys())}")
        
        # Test parameter learning
        processor = create_dasp_ace_processor(sample_rate)
        print(f"✓ Created processor with {sum(p.numel() for p in processor.parameters())} parameters")
        
        # Show that gradients work
        processor.train()
        output = processor(test_audio)['output']
        loss = torch.mean(output ** 2)  # Dummy loss
        loss.backward()
        
        print(f"✓ Gradients computed successfully!")
        print(f"Loss: {loss.item():.6f}")
        
        print("\n🎉 Optimized DASP ACE is working!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

def create_ace_with_gui_params(sample_rate: float = 44100) -> OptimizedDASPACE:
    ace = OptimizedDASPACE(sample_rate=sample_rate)

    gui_params = {
        # TCE
        "fHPFtc":      4000.0,
        "tauAtc":      7.0,
        "tauDtc":      16.0,
        "nu":         -60.0,

        # High-shelf
        "fHSF":        4000.0,
        "sHSF":           0.1,
        "dbHSF":         0.0,

        # Noise
        "dbNoise":    -96.0,

        # LI
        "rho":         25.0,
        "tauLI":       7.0,

        # EX
        "beta":        1.0,
        "mu":         -3.0,
        "tauEX":       7.0,

        # DP
        "tauAdp":      7.0,
        "t60atCutoff": 0.72,   # seconds, as in GUI
        "dpCutoff":  1000.0,

        # Regularization
        "dBregDelta": -96.0,

        # Summation smoothing
        "tauSP":       2.0,

        # Mixing
        "wet":         0.9,
        "dimWeight":   1.0,    # SCE only
        "vol":         6.0,    # dB
    }

    with torch.no_grad():
        for name, value in gui_params.items():
            if hasattr(ace, name):
                getattr(ace, name).data.fill_(float(value))

        # band_gains = 1.0 for all bands (GUI default)
        ace.band_gains.data.fill_(1.0)

    return ace
