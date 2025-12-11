#!/usr/bin/env python3
"""
Offline Audio Processing with DASP ACE
Simple script to enhance your audio files using the optimized DASP ACE implementation
"""

import torch
import torchaudio
import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    # from ace_dasp_optimized import OptimizedDASPACE
    from ace_dasp import OptimizedDASPACE, create_ace_with_gui_params
    print("✓ DASP ACE implementation loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import DASP ACE: {e}")
    print("Make sure ace_dasp_optimized.py is in the same directory")
    sys.exit(1)


def calculate_snr_metrics(original: torch.Tensor, enhanced: torch.Tensor, noise_ref: torch.Tensor = None) -> dict:
    """
    Calculate various SNR metrics for audio enhancement evaluation
    
    Args:
        original: Original input signal [channels, time]
        enhanced: Enhanced output signal [channels, time]
        noise_ref: Optional reference noise signal for absolute SNR
    
    Returns:
        Dictionary with SNR measurements
    """
    eps = 1e-10  # Small value to avoid log(0)
    
    # Ensure signals have same length
    min_length = min(original.shape[-1], enhanced.shape[-1])
    original = original[..., :min_length]
    enhanced = enhanced[..., :min_length]
    
    # Calculate power metrics
    original_power = torch.mean(original ** 2, dim=-1, keepdim=True)
    enhanced_power = torch.mean(enhanced ** 2, dim=-1, keepdim=True)
    
    # Signal-to-Noise Ratio calculations
    snr_metrics = {}
    
    # 1. Power SNR (ratio of signal powers)
    power_ratio = enhanced_power / (original_power + eps)
    snr_metrics['power_snr_db'] = 10 * torch.log10(power_ratio + eps).mean().item()
    
    # 2. RMS SNR  
    original_rms = torch.sqrt(original_power)
    enhanced_rms = torch.sqrt(enhanced_power)
    rms_ratio = enhanced_rms / (original_rms + eps)
    snr_metrics['rms_snr_db'] = 20 * torch.log10(rms_ratio + eps).mean().item()
    
    # 3. Difference-based SNR (enhancement vs difference)
    difference = enhanced - original
    difference_power = torch.mean(difference ** 2, dim=-1, keepdim=True)
    enhancement_snr = enhanced_power / (difference_power + eps)
    snr_metrics['enhancement_snr_db'] = 10 * torch.log10(enhancement_snr + eps).mean().item()
    
    # 4. Segmental SNR (frame-by-frame analysis)
    frame_size = 1024
    num_frames = min_length // frame_size
    if num_frames > 0:
        seg_snrs = []
        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = (i + 1) * frame_size
            
            orig_frame = original[..., start_idx:end_idx]
            enh_frame = enhanced[..., start_idx:end_idx]
            
            orig_frame_power = torch.mean(orig_frame ** 2)
            enh_frame_power = torch.mean(enh_frame ** 2)
            
            if orig_frame_power > eps:
                frame_snr = 10 * torch.log10((enh_frame_power + eps) / (orig_frame_power + eps))
                seg_snrs.append(frame_snr.item())
        
        if seg_snrs:
            snr_metrics['segmental_snr_db'] = sum(seg_snrs) / len(seg_snrs)
            snr_metrics['segmental_snr_std'] = torch.tensor(seg_snrs).std().item()
    
    # 5. Dynamic range metrics
    original_peak = torch.max(torch.abs(original))
    enhanced_peak = torch.max(torch.abs(enhanced))
    original_rms_total = torch.sqrt(torch.mean(original ** 2))
    enhanced_rms_total = torch.sqrt(torch.mean(enhanced ** 2))
    
    snr_metrics['original_dynamic_range_db'] = 20 * torch.log10((original_peak + eps) / (original_rms_total + eps)).item()
    snr_metrics['enhanced_dynamic_range_db'] = 20 * torch.log10((enhanced_peak + eps) / (enhanced_rms_total + eps)).item()
    
    # 6. Crest factor (peak to RMS ratio)
    snr_metrics['original_crest_factor_db'] = 20 * torch.log10((original_peak + eps) / (original_rms_total + eps)).item()
    snr_metrics['enhanced_crest_factor_db'] = 20 * torch.log10((enhanced_peak + eps) / (enhanced_rms_total + eps)).item()
    
    # 7. Absolute levels
    snr_metrics['original_rms_db'] = 20 * torch.log10(original_rms_total + eps).item()
    snr_metrics['enhanced_rms_db'] = 20 * torch.log10(enhanced_rms_total + eps).item()
    snr_metrics['original_peak_db'] = 20 * torch.log10(original_peak + eps).item()
    snr_metrics['enhanced_peak_db'] = 20 * torch.log10(enhanced_peak + eps).item()
    
    # 8. If noise reference is provided, calculate absolute SNR
    if noise_ref is not None:
        noise_power = torch.mean(noise_ref ** 2)
        original_snr_abs = original_power.mean() / (noise_power + eps)
        enhanced_snr_abs = enhanced_power.mean() / (noise_power + eps)
        
        snr_metrics['original_snr_absolute_db'] = 10 * torch.log10(original_snr_abs + eps).item()
        snr_metrics['enhanced_snr_absolute_db'] = 10 * torch.log10(enhanced_snr_abs + eps).item()
        snr_metrics['snr_improvement_db'] = snr_metrics['enhanced_snr_absolute_db'] - snr_metrics['original_snr_absolute_db']
    
    return snr_metrics


def process_audio_file(input_path: str, 
                      output_path: str,
                      sample_rate: int = 44100,
                      n_bands: int = 32,
                      ace_params: dict = None) -> dict:
    """
    Process a single audio file with DASP ACE
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save enhanced audio
        sample_rate: Target sample rate (will resample if needed)
        n_bands: Number of frequency bands for processing
        ace_params: Optional parameter overrides
    
    Returns:
        Dictionary with processing info and statistics
    """
    
    print(f"Processing: {input_path}")
    
    # Load audio
    try:
        waveform, orig_sr = torchaudio.load(input_path)
        print(f"✓ Loaded audio: {waveform.shape}, {orig_sr} Hz")
    except Exception as e:
        print(f"✗ Failed to load audio: {e}")
        return None
    
    # Resample if needed
    if orig_sr != sample_rate:
        print(f"Resampling from {orig_sr} Hz to {sample_rate} Hz...")
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)
    
    # Ensure stereo (duplicate mono to stereo if needed)
    if waveform.shape[0] == 1:
        print("Converting mono to stereo...")
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        print("Converting multi-channel to stereo...")
        waveform = waveform[:2]  # Take first 2 channels
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0)  # [1, channels, time]
    
    # Create ACE processor
    print(f"Initializing DASP ACE processor ({n_bands} bands)...")
    # ace_processor = OptimizedDASPACE(
    #     sample_rate=float(sample_rate),
    #     n_bands=n_bands,
    #     filter_duration=0.02
    # )
    ace_processor = create_ace_with_gui_params(sample_rate=int(sample_rate))
    
    # Set custom parameters if provided
    if ace_params:
        print("Applying custom parameters...")
        with torch.no_grad():
            for param_name, value in ace_params.items():
                if hasattr(ace_processor, param_name):
                    getattr(ace_processor, param_name).data.fill_(value)
                    print(f"  {param_name}: {value}")
                else:
                    print(f"  Warning: Parameter '{param_name}' not found")
    
    # Process audio (no grad mode for inference)
    print("Processing audio...")
    ace_processor.eval()
    
    with torch.no_grad():
        results = ace_processor(waveform)
    
    enhanced_audio = results['output'].squeeze(0)  # Remove batch dimension
    
    # Calculate SNR metrics
    print("Calculating SNR metrics...")
    snr_metrics = calculate_snr_metrics(
        waveform.squeeze(0),  # Remove batch dimension for SNR calculation
        enhanced_audio
    )
    
    # Calculate statistics
    input_rms = torch.sqrt(torch.mean(waveform ** 2)).item()
    output_rms = torch.sqrt(torch.mean(enhanced_audio ** 2)).item()
    enhancement_ratio = output_rms / input_rms if input_rms > 0 else 1.0
    
    # Save enhanced audio
    print(f"Saving enhanced audio to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there is one
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        torchaudio.save(output_path, enhanced_audio, sample_rate)
        print("✓ Audio saved successfully")
    except Exception as e:
        print(f"✗ Failed to save audio: {e}")
        return None
    
    # Return processing info
    duration = waveform.shape[-1] / sample_rate
    return {
        'input_path': input_path,
        'output_path': output_path,
        'input_shape': tuple(waveform.squeeze(0).shape),
        'output_shape': tuple(enhanced_audio.shape),
        'duration': duration,
        'sample_rate': sample_rate,
        'n_bands': n_bands,
        'input_rms': input_rms,
        'output_rms': output_rms,
        'enhancement_ratio': enhancement_ratio,
        'snr_metrics': snr_metrics,
        'parameters': {name: param.item() if param.numel() == 1 else param.mean().item() 
                      for name, param in ace_processor.named_parameters()}
    }


def main():
    """Main command line interface"""
    parser = argparse.ArgumentParser(description='Enhance audio files with DASP ACE')
    parser.add_argument('input', help='Input audio file path')
    parser.add_argument('-o', '--output', help='Output audio file path (default: input_enhanced.wav)')
    parser.add_argument('-sr', '--sample-rate', type=int, default=44100, help='Sample rate (default: 44100)')
    parser.add_argument('-b', '--bands', type=int, default=32, help='Number of frequency bands (default: 32)')
    parser.add_argument('--tce-gain', type=float, help='TCE gain parameter')
    parser.add_argument('--tce-threshold', type=float, help='TCE threshold parameter')
    parser.add_argument('--sce-gain', type=float, help='SCE gain parameter')
    parser.add_argument('--output-gain', type=float, help='Output gain parameter')
    parser.add_argument('--wet-dry-mix', type=float, help='Wet/dry mix (0.0-1.0)')
    parser.add_argument('--attack-time', type=float, help='Attack time in seconds')
    parser.add_argument('--decay-time', type=float, help='Decay time in seconds')
    parser.add_argument('--vol', type=float, help='Master volume in dB')
    parser.add_argument('--wet', type=float, help='Wet/dry mix amount (0.0-1.0)')
    parser.add_argument('--dim-weight', type=float, help='Spectral vs temporal weight (0.0-1.0)')
    parser.add_argument('--snr-only', action='store_true', help='Only calculate and display SNR metrics without processing')
    parser.add_argument('--noise-ref', type=str, help='Reference noise file for absolute SNR calculation')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"✗ Input file not found: {args.input}")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}")
    
    # Collect custom parameters
    ace_params = {}
    if args.tce_gain is not None:
        ace_params['tce_gain'] = args.tce_gain
    if args.tce_threshold is not None:
        ace_params['tce_threshold'] = args.tce_threshold
    if args.sce_gain is not None:
        ace_params['sce_gain'] = args.sce_gain
    if args.output_gain is not None:
        ace_params['output_gain'] = args.output_gain
    if args.wet_dry_mix is not None:
        ace_params['wet_dry_mix'] = args.wet_dry_mix
    if args.attack_time is not None:
        ace_params['attack_time'] = args.attack_time
    if args.decay_time is not None:
        ace_params['decay_time'] = args.decay_time
    if args.vol is not None:
        ace_params['vol'] = args.vol
    if args.wet is not None:
        ace_params['wet'] = args.wet
    if args.dim_weight is not None:
        ace_params['dimWeight'] = args.dim_weight
    
    print("=" * 60)
    print("🎵 DASP ACE Offline Audio Processing 🎵")
    print("=" * 60)
    
    # SNR-only mode
    if args.snr_only:
        print("📊 SNR Analysis Mode - No processing, just measurement")
        
        # Load original audio
        try:
            waveform, sample_rate = torchaudio.load(args.input)
            print(f"✓ Loaded audio: {waveform.shape}, {sample_rate} Hz")
        except Exception as e:
            print(f"✗ Failed to load audio: {e}")
            sys.exit(1)
        
        # Load noise reference if provided
        noise_ref = None
        if args.noise_ref:
            try:
                noise_waveform, noise_sr = torchaudio.load(args.noise_ref)
                if noise_sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(noise_sr, sample_rate)
                    noise_waveform = resampler(noise_waveform)
                noise_ref = noise_waveform
                print(f"✓ Loaded noise reference: {noise_waveform.shape}")
            except Exception as e:
                print(f"⚠️ Failed to load noise reference: {e}")
        
        # Calculate SNR metrics
        snr_metrics = calculate_snr_metrics(waveform, waveform, noise_ref)
        
        print("\n📊 SIGNAL ANALYSIS RESULTS")
        print("=" * 40)
        print(f"File: {args.input}")
        print(f"Duration: {waveform.shape[-1] / sample_rate:.2f} seconds")
        print(f"Channels: {waveform.shape[0]}")
        
        print(f"\n🔊 Signal Levels:")
        print(f"RMS Level:           {snr_metrics['original_rms_db']:.2f} dB")
        print(f"Peak Level:          {snr_metrics['original_peak_db']:.2f} dB")
        print(f"Dynamic Range:       {snr_metrics['original_dynamic_range_db']:.2f} dB")
        print(f"Crest Factor:        {snr_metrics['original_crest_factor_db']:.2f} dB")
        
        if 'original_snr_absolute_db' in snr_metrics:
            print(f"Absolute SNR:        {snr_metrics['original_snr_absolute_db']:.2f} dB")
        
        return
    
    # Normal processing mode
    result = process_audio_file(
        args.input,
        output_path,
        sample_rate=args.sample_rate,
        n_bands=args.bands,
        ace_params=ace_params
    )
    
    if result:
        print("\n" + "=" * 60)
        print("📊 PROCESSING RESULTS")
        print("=" * 60)
        print(f"Input file:      {result['input_path']}")
        print(f"Output file:     {result['output_path']}")
        print(f"Duration:        {result['duration']:.2f} seconds")
        print(f"Sample rate:     {result['sample_rate']} Hz")
        print(f"Frequency bands: {result['n_bands']}")
        print(f"Input RMS:       {result['input_rms']:.4f}")
        print(f"Output RMS:      {result['output_rms']:.4f}")
        print(f"Enhancement:     {result['enhancement_ratio']:.2f}x")
        
        # Display SNR metrics
        print(f"\n📊 SNR Analysis:")
        snr = result['snr_metrics']
        print(f"Power SNR:           {snr['power_snr_db']:.2f} dB")
        print(f"RMS SNR:             {snr['rms_snr_db']:.2f} dB")
        print(f"Enhancement SNR:     {snr['enhancement_snr_db']:.2f} dB")
        
        if 'segmental_snr_db' in snr:
            print(f"Segmental SNR:       {snr['segmental_snr_db']:.2f} ± {snr['segmental_snr_std']:.2f} dB")
        
        print(f"\n📈 Dynamic Range:")
        print(f"Original DR:         {snr['original_dynamic_range_db']:.2f} dB")
        print(f"Enhanced DR:         {snr['enhanced_dynamic_range_db']:.2f} dB")
        print(f"Original Crest:      {snr['original_crest_factor_db']:.2f} dB")
        print(f"Enhanced Crest:      {snr['enhanced_crest_factor_db']:.2f} dB")
        
        print(f"\n🔊 Signal Levels:")
        print(f"Original RMS:        {snr['original_rms_db']:.2f} dB")
        print(f"Enhanced RMS:        {snr['enhanced_rms_db']:.2f} dB")
        print(f"Original Peak:       {snr['original_peak_db']:.2f} dB")
        print(f"Enhanced Peak:       {snr['enhanced_peak_db']:.2f} dB")
        
        print(f"\nACE Parameters:")
        for name, value in result['parameters'].items():
            print(f"  {name}: {value:.4f}")
        
        print(f"\n✅ Processing completed successfully!")
        print(f"Enhanced audio saved to: {output_path}")
    else:
        print("\n❌ Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Interactive mode for testing
        print("🎵 DASP ACE Offline Audio Processor 🎵")
        print("=" * 50)
        print("Interactive mode - Testing with built-in audio")
        print()
        
        # Create a test audio file if none provided
        test_audio_path = "wav/test_audio.wav"
        enhanced_path = "wav/test_audio_enhanced.wav"
        
        print("Generating test audio...")
        sample_rate = 44100
        duration = 3.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Create a complex test signal
        signal = (
            torch.sin(2 * torch.pi * 220 * t) +           # A3
            0.5 * torch.sin(2 * torch.pi * 440 * t) +     # A4
            0.3 * torch.sin(2 * torch.pi * 880 * t) +     # A5
            0.1 * torch.randn_like(t)                     # Noise
        )
        
        # Make stereo
        stereo_signal = torch.stack([signal, signal * 0.9])
        
        # Save test audio
        torchaudio.save(test_audio_path, stereo_signal, sample_rate)
        print(f"✓ Test audio created: {test_audio_path}")
        
        # Process it
        result = process_audio_file(
            test_audio_path,
            enhanced_path,
            sample_rate=sample_rate,
            n_bands=24,  # Fewer bands for faster processing
            ace_params={
                'wet': 0.8,     # Use SuperCollider parameter names
                'vol': 3.0,     # Volume in dB
                'dimWeight': 0.7  # Spectral vs temporal weight
            }
        )
        
        if result:
            print(f"\n🎉 Test completed! Listen to '{enhanced_path}' to hear the ACE enhancement.")
            
            # # Clean up test file
            # if os.path.exists(test_audio_path):
            #     os.remove(test_audio_path)
                
        print(f"\nTo process your own audio files, run:")
        print(f"python {sys.argv[0]} your_audio.wav")
        print(f"python {sys.argv[0]} your_audio.wav -o enhanced_audio.wav --tce-gain 2.5")