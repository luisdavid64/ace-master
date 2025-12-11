"""
Optimized ACE Implementation using DASP PyTorch - Version 2
Focus on using actual DASP components and efficient processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import dasp_pytorch.signal_dasp as dasp_signal
import dasp_pytorch.functional as dasp_F


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
        self.filterbank = dasp_signal.gammatone_filterbank(
            sample_rate=int(sample_rate),
            num_bands=n_bands,
            low_freq=50.0,
            high_freq=sample_rate // 2,
            order=4,
            duration=filter_duration
        )
        
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

        
    """ Temporal Contrast Enhancement (TCE) Functions """

    def _time_constant_to_alpha(self, tau_ms: torch.Tensor) -> torch.Tensor:
        """
        Convert time constant tau (in ms) to smoothing factor alpha (Eq. 10).
        tau_ms: scalar tensor in milliseconds
        """
        # tau in seconds
        tau_s = torch.clamp(tau_ms, 0.1, 1000.0) / 1000.0
        fs = torch.tensor(float(self.sample_rate), device=tau_ms.device)
        # Eq. (10): alpha = exp(-1 / (tau * fs))
        alpha = torch.exp(-1.0 / (tau_s * fs))
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
        alpha = self._time_constant_to_alpha(tau_ms)
        return self._nonlinear_envelope(x, alpha, mode="attack")

    def envd(self, x: torch.Tensor, tau_ms: torch.Tensor) -> torch.Tensor:
        """Envelope with smooth decay, instant attack."""
        alpha = self._time_constant_to_alpha(tau_ms)
        return self._nonlinear_envelope(x, alpha, mode="decay")

    
    def high_pass_filter(self, x: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """Simple high-pass filter implementation"""
        # For now, use a simple frequency-domain approach
        # In production, you'd implement proper biquad filtering
        return x  # Placeholder
    
    def high_shelf_filter(self, x: torch.Tensor, freq: torch.Tensor, slope: torch.Tensor, gain_db: torch.Tensor) -> torch.Tensor:
        """High shelf filter implementation"""
        gain_linear = torch.pow(10.0, gain_db / 20.0)
        return x * gain_linear  # Simplified implementation
    
    def db_to_amplitude(self, db: torch.Tensor) -> torch.Tensor:
        """Convert dB to linear amplitude"""
        return torch.pow(10.0, db / 20.0)
    
    def temporal_contrast_enhancement(self, x: torch.Tensor) -> torch.Tensor:
        """
        Temporal Contrast Enhancement (TCE) following Weger et al. (ICAD 2019), Sec. 2.2, Eqs. (12–14).

        x: [batch, channels, time]  -> original signal s[n]
        returns st[n] with transient emphasis
        """

        # 1) High-pass filter to emphasize high-frequency transients
        # In the paper, this is a 2nd-order HPF; here we call a placeholder
        # you should implement this with a proper biquad or DASP filter.
        tce_hpf_freq = torch.clamp(self.fHPFtc, 100.0, self.sample_rate / 2.0)
        sh = self.high_pass_filter(x, tce_hpf_freq)  # sh[n]

        # 2) Envelope followers for transient detection
        # Time constants for temporal ACE (paper suggests tauA ≈ 3 ms, tauD ≈ 7 ms)
        tau_a_ms = torch.clamp(self.tauAtc, 1.0, 100.0)  # attack time constant for enva
        tau_d_ms = torch.clamp(self.tauDtc, 1.0, 100.0)  # decay time constant for envd

        # et,d[n] = envd{ sh[n] }
        et_d = self.envd(sh, tau_d_ms)

        # et,a[n] = enva{ et,d[n] }
        et_a = self.enva(et_d, tau_a_ms)

        # 3) Transient envelope et[n] = max{ et,d - et,a - nu, 0 }
        # Your parameter self.nu is stored in dB; convert to linear amplitude.
        # Clamp like in the paper (roughly -60..0 dB relative threshold).
        nu_amp = self.db_to_amplitude(torch.clamp(self.nu, -60.0, 0.0))
        # Broadcast to match [B, C, T]
        nu_amp = nu_amp.to(x.device, x.dtype).view(1, 1, 1)

        et = (et_d - et_a - nu_amp).clamp(min=0.0)

        # 4) Normalization envelope envd{et[n]} for equation (14)
        # Use the same decay time constant tau_d_ms for this envd.
        et_env = self.envd(et, tau_d_ms)

        # Regularization: add small delta (you already have dBregDelta)
        reg_delta = self.db_to_amplitude(torch.clamp(self.dBregDelta, -120.0, -60.0))
        reg_delta = reg_delta.to(x.device, x.dtype).view(1, 1, 1)

        denom = et_env + reg_delta

        # 5) Output: st[n] = s[n] * et[n] / envd{et[n]}
        modulation = et / denom
        st = x * modulation

        return st
    
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

    def spectral_contrast_enhancement(self,
                                      band_signals: torch.Tensor) -> torch.Tensor:
        """
        Spectral ACE: LI (sharpening) + EX (dynamics expansion)
        + DP (decay prolongation), following Weger et al. (ICAD 2019).

        band_signals: [B, C, K, T]  (real subband signals ck[n])
        returns processed subbands c'_k[n] with envelopes pk[n].
        """
        assert band_signals.dim() == 4
        B, C, K, T = band_signals.shape
        x = band_signals

        device, dtype = x.device, x.dtype

        # 0) Envelopes ek[n]
        ek = x.abs()                                         # [B,C,K,T]

        # 1) Smooth envelopes for LI (ẽk)  -- τ ≈ 7 ms in paper
        tau_li_ms = torch.clamp(self.tauLI, 1.0, 50.0)
        e_tilde = self.leaky_integrator(ek, tau_li_ms)       # [B,C,K,T]

        # 2) Lateral inhibition term Tk[n]
        Tk = self._lateral_inhibition(e_tilde)               # [B,C,K,T]

        # 3) Spectral sharpening (Eq. 6)
        rho = torch.clamp(self.rho, 0.0, 100.0)
        reg_delta = self.db_to_amplitude(
            torch.clamp(self.dBregDelta, -120.0, -60.0)
        ).to(device=device, dtype=dtype).view(1, 1, 1, 1)

        ratio_li = torch.pow(
            torch.clamp(e_tilde / (Tk + reg_delta), min=0.0),
            rho
        )
        ratio_li = ratio_li.clamp(max=1.0)
        uk = ek * ratio_li                                   # [B,C,K,T]

        # 4) Smoothed envelopes for EX (ũk) with τ_EX
        tau_ex_ms = torch.clamp(self.tauEX, 1.0, 100.0)
        u_tilde = self.leaky_integrator(uk, tau_ex_ms)       # [B,C,K,T]

        # Instantaneous global max û_max[n] (Eq. 8)
        u_max, _ = u_tilde.max(dim=2, keepdim=True)          # [B,C,1,T]

        # 5) Spectral dynamics expansion (Eq. 7)
        beta = torch.clamp(self.beta, 0.0, 20.0)
        # mu stored in dB -> convert to linear in [0,1]
        mu_lin = torch.clamp(self.db_to_amplitude(self.mu),
                             0.0, 1.0).to(
                                 device=device, dtype=dtype
                             ).view(1, 1, 1, 1)

        thresh = mu_lin * u_max                              # [B,C,1,T]

        gain1 = torch.pow(
            torch.clamp(u_tilde / (thresh + reg_delta), min=0.0),
            beta
        )
        gain2 = torch.clamp(u_max / (u_tilde + reg_delta), min=0.0)

        gain_ex = torch.minimum(gain1, gain2)                # Eq. 7 min(...)
        vk = uk * gain_ex                                    # [B,C,K,T]

        # 6) Decay prolongation (Eq. 11)
        band_centers = self._erb_space(50.0,
                                       self.sample_rate / 2.0,
                                       K, device, dtype)     # [K]
        pk = self._decay_prolongation(vk, band_centers)      # [B,C,K,T]

        # 7) Final short smoothing (LP blocks before Eq. 2)
        tau_sp_ms = torch.clamp(self.tauSP, 0.5, 20.0)       # paper uses ≈2 ms
        ek_lp = self.leaky_integrator(ek, tau_sp_ms)
        pk_lp = self.leaky_integrator(pk, tau_sp_ms)

        # 8) Apply processed envelope back to subbands (Eq. 2)
        env_ratio = pk_lp / (ek_lp + reg_delta)

        # Optional per-band weights
        band_weights = torch.clamp(self.band_gains,
                                   0.0, 5.0).view(1, 1, K, 1)

        c_out = x * env_ratio * band_weights                 # [B,C,K,T]

        return c_out
 
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete ACE processing pipeline following SuperCollider implementation
        """
        # Store original for wet/dry mix
        dry = x.clone()
        
        # Input high-pass filtering
        hpf_freq = torch.clamp(self.fHPF, 20.0, 1000.0)
        filtered_input = self.high_pass_filter(x, hpf_freq)
        
        # Temporal Contrast Enhancement (TCE)
        tce_output = self.temporal_contrast_enhancement(filtered_input)
        
        # Spectral Contrast Enhancement (SCE)
        # Apply gammatone filterbank
        band_signals = self.apply_gammatone_filterbank(filtered_input)
        
        # Apply SCE processing
        sce_bands = self.spectral_contrast_enhancement(band_signals)
        
        # Reconstruct signal by summing across frequency bands
        noise_amp = self.db_to_amplitude(
            torch.clamp(self.dbNoise, -120.0, -60.0)
        )
        noisy_input = filtered_input + noise_amp * torch.randn_like(filtered_input)

        band_signals = self.apply_gammatone_filterbank(noisy_input)   # [B,C,K,T]

        sce_bands = self.spectral_contrast_enhancement(band_signals)  # [B,C,K,T]
        sce_output = torch.sum(sce_bands, dim=2)                       # [B,C,T]
        # sce_output = torch.sum(sce_bands, dim=2)  # [batch, channels, time]
        
        # Apply high shelf filter
        hsf_freq = torch.clamp(self.fHSF, 1000.0, self.sample_rate/2)
        hsf_slope = torch.clamp(self.sHSF, 0.1, 2.0)
        hsf_gain_neg = -torch.clamp(self.dbHSF, -20.0, 20.0)  # Negative for compensation
        sce_filtered = self.high_shelf_filter(sce_output, hsf_freq, hsf_slope, hsf_gain_neg)
        
        # Crossfade between TCE and SCE based on dimWeight
        dim_weight = torch.clamp(self.dimWeight, 0.0, 1.0)
        enhanced = (1 - dim_weight) * tce_output + dim_weight * sce_filtered
        
        # Final high-pass filter
        enhanced = self.high_pass_filter(enhanced, hpf_freq)
        
        # Output processing: volume and wet/dry mix
        vol_gain = self.db_to_amplitude(torch.clamp(self.vol, -60.0, 20.0))
        wet_amount = torch.clamp(self.wet, 0.0, 1.0)
        
        # Apply volume
        wet = enhanced * vol_gain
        
        # Wet/dry mix
        output = (1 - wet_amount) * dry + wet_amount * wet
        
        # Clip output to prevent clipping
        output = torch.clamp(output, -1.0, 1.0)
        
        return {
            'output': output,
            'dry': dry,
            'wet': wet,
            'enhanced': enhanced,
            'tce_output': tce_output,
            'sce_output': sce_output,
            'band_signals': band_signals,
            'sce_bands': sce_bands
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