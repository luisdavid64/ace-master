#!/usr/bin/env python3
"""
Training-Optimized ACE Implementation
Maintains authentic SuperCollider sound while optimizing for ML training speed
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional
from ace_dasp import OptimizedDASPACE, create_ace_with_gui_params

class FastACE(OptimizedDASPACE):
    """
    Training-optimized version of ACE with speed improvements for ML training
    while preserving authentic SuperCollider sound quality
    """
    
    def __init__(self, sample_rate: float = 44100, n_bands: int = 32, 
                 training_mode: bool = True, chunk_size: Optional[int] = None):
        
        self.training_mode = training_mode
        self.chunk_size = chunk_size or (8192 if training_mode else None)
        
        # Initialize with optimizations
        super().__init__(sample_rate, n_bands)
        
        # Training-specific optimizations
        self._setup_training_optimizations()
    
    def _setup_training_optimizations(self):
        """Setup optimizations specific to training"""
        
        # 1. Cache frequently used values
        self.register_buffer('_sqrt_2pi', torch.tensor(math.sqrt(2 * math.pi)))
        self.register_buffer('_log10_cache', torch.log(torch.tensor(10.0)))
        
        # 2. Pre-compute time constants that don't change often
        self._cached_alphas = {}
        
        # 3. Setup for mixed precision training
        self.use_amp = True  # Automatic Mixed Precision
        
        print(f"FastACE optimizations: chunk_size={self.chunk_size}, AMP={self.use_amp}")
    
    def forward(self, x: torch.Tensor, use_checkpointing: bool = False) -> Dict[str, torch.Tensor]:
        """
        Optimized forward pass with optional gradient checkpointing
        """
        if self.training_mode and self.chunk_size and x.shape[-1] > self.chunk_size:
            return self._chunked_forward(x, use_checkpointing)
        else:
            return self._standard_forward(x, use_checkpointing)
    
    def _chunked_forward(self, x: torch.Tensor, use_checkpointing: bool = False) -> Dict[str, torch.Tensor]:
        """
        Process long audio in chunks to reduce memory usage during training
        """
        B, C, T = x.shape
        chunk_size = self.chunk_size
        
        # Split into overlapping chunks to avoid boundary artifacts
        overlap = min(1024, chunk_size // 4)  # 25% overlap
        step = chunk_size - overlap
        
        chunks = []
        for i in range(0, T - overlap, step):
            end_idx = min(i + chunk_size, T)
            chunk = x[:, :, i:end_idx]
            chunks.append(chunk)
        
        # Process chunks
        chunk_outputs = []
        for chunk in chunks:
            if use_checkpointing:
                chunk_out = torch.utils.checkpoint.checkpoint(
                    self._process_chunk, chunk, use_reentrant=False
                )
            else:
                chunk_out = self._process_chunk(chunk)
            chunk_outputs.append(chunk_out)
        
        # Merge chunks with overlap handling
        return self._merge_chunks(chunk_outputs, overlap, T)
    
    def _process_chunk(self, chunk: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a single chunk"""
        return self._standard_forward(chunk, use_checkpointing=False)
    
    def _merge_chunks(self, chunk_outputs: list, overlap: int, target_length: int) -> Dict[str, torch.Tensor]:
        """Merge overlapping chunks with crossfading"""
        if len(chunk_outputs) == 1:
            return chunk_outputs[0]
        
        # Get output keys from first chunk
        keys = chunk_outputs[0].keys()
        merged = {}
        
        for key in keys:
            chunks = [output[key] for output in chunk_outputs]
            merged_tensor = self._crossfade_chunks(chunks, overlap, target_length)
            merged[key] = merged_tensor
        
        return merged
    
    def _crossfade_chunks(self, chunks: list, overlap: int, target_length: int) -> torch.Tensor:
        """Crossfade overlapping chunks"""
        if len(chunks) == 1:
            return chunks[0]
        
        B, C = chunks[0].shape[:2]
        step = chunks[0].shape[-1] - overlap
        
        # Initialize output
        output = torch.zeros(B, C, target_length, device=chunks[0].device, dtype=chunks[0].dtype)
        
        # Place first chunk
        output[:, :, :chunks[0].shape[-1]] = chunks[0]
        
        # Crossfade subsequent chunks
        for i, chunk in enumerate(chunks[1:], 1):
            start_idx = i * step
            end_idx = start_idx + chunk.shape[-1]
            
            if end_idx > target_length:
                chunk = chunk[:, :, :target_length - start_idx]
                end_idx = target_length
            
            if start_idx < target_length:
                # Crossfade in overlap region
                if overlap > 0 and start_idx >= overlap:
                    fade_start = start_idx
                    fade_end = min(fade_start + overlap, end_idx, target_length)
                    fade_length = fade_end - fade_start
                    
                    if fade_length > 0:
                        # Linear crossfade
                        fade_in = torch.linspace(0, 1, fade_length, device=chunk.device)
                        fade_out = 1 - fade_in
                        
                        # Apply crossfade
                        chunk_part = chunk[:, :, :fade_length]
                        existing_part = output[:, :, fade_start:fade_end]
                        
                        output[:, :, fade_start:fade_end] = (
                            existing_part * fade_out + chunk_part * fade_in
                        )
                        
                        # Add non-overlapping part
                        if fade_end < end_idx:
                            non_overlap_chunk = chunk[:, :, fade_length:]
                            output[:, :, fade_end:end_idx] = non_overlap_chunk
                    else:
                        output[:, :, start_idx:end_idx] = chunk
                else:
                    output[:, :, start_idx:end_idx] = chunk
        
        return output
    
    def _standard_forward(self, x: torch.Tensor, use_checkpointing: bool = False) -> Dict[str, torch.Tensor]:
        """Standard forward pass with optional checkpointing"""
        
        # Use automatic mixed precision if enabled
        if self.use_amp and self.training:
            with torch.cuda.amp.autocast():
                return super().forward(x)
        else:
            return super().forward(x)
    
    def fast_amplitude_env(self, x: torch.Tensor, attack_ms: torch.Tensor, 
                          decay_ms: torch.Tensor) -> torch.Tensor:
        """
        Optimized amplitude envelope follower with caching
        """
        # Cache key based on time constants
        cache_key = (attack_ms.item(), decay_ms.item())
        
        if cache_key in self._cached_alphas:
            alphaA, alphaR = self._cached_alphas[cache_key]
        else:
            alphaA = self._time_constant_to_alpha(attack_ms)
            alphaR = self._time_constant_to_alpha(decay_ms)
            self._cached_alphas[cache_key] = (alphaA, alphaR)
        
        # Vectorized envelope following
        return self._vectorized_envelope_follower(x, alphaA, alphaR)
    
    def _vectorized_envelope_follower(self, x: torch.Tensor, alphaA: torch.Tensor, 
                                    alphaR: torch.Tensor) -> torch.Tensor:
        """
        Optimized vectorized envelope follower
        """
        # Use unfold for efficient sliding window processing
        if x.shape[-1] > 1024:  # Only for longer signals
            return self._efficient_envelope_unfold(x, alphaA, alphaR)
        else:
            return self.amplitude_env(x, alphaA, alphaR)  # Use original for short signals
    
    def _efficient_envelope_unfold(self, x: torch.Tensor, alphaA: torch.Tensor, 
                                 alphaR: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient envelope follower using unfold
        """
        # This is a simplified version - the full implementation would be more complex
        # For now, fallback to original
        return self.amplitude_env(x, alphaA, alphaR)
    
    @torch.jit.script_method if hasattr(torch.jit, 'script_method') else lambda x: x
    def fast_db_to_amplitude(self, db: torch.Tensor) -> torch.Tensor:
        """JIT-compiled dB to amplitude conversion"""
        return torch.pow(10.0, db / 20.0)
    
    def enable_compilation(self):
        """Enable PyTorch 2.0 compilation for speed"""
        try:
            if hasattr(torch, 'compile'):
                print("Enabling torch.compile for speed...")
                # Compile key methods
                self._standard_forward = torch.compile(self._standard_forward)
                self.gammatone_filter = torch.compile(self.gammatone_filter)
                print("✅ Compilation enabled")
            else:
                print("PyTorch 2.0+ required for compilation")
        except Exception as e:
            print(f"Compilation failed: {e}")


def create_fast_ace(sample_rate: float = 44100, training_mode: bool = True,
                   chunk_size: int = 8192, enable_compile: bool = True) -> FastACE:
    """
    Create a training-optimized ACE with SuperCollider parameters
    """
    # Create fast ACE
    ace = FastACE(sample_rate=sample_rate, training_mode=training_mode, 
                  chunk_size=chunk_size)
    
    # Apply SuperCollider GUI parameters
    gui_params = {
        "fHPFtc":      4000.0,
        "tauAtc":      7.0,
        "tauDtc":      16.0,
        "nu":         -60.0,
        "fHSF":        4000.0,
        "sHSF":           0.1,
        "dbHSF":         0.0,
        "dbNoise":    -96.0,
        "rho":         25.0,
        "tauLI":       7.0,
        "beta":        1.0,
        "mu":         -3.0,
        "tauEX":       7.0,
        "tauAdp":      7.0,
        "t60atCutoff": 0.72,
        "dpCutoff":  1000.0,
        "dBregDelta": -96.0,
        "tauSP":       2.0,
        "wet":         0.9,
        "dimWeight":   1.0,
        "vol":         6.0,
    }
    
    with torch.no_grad():
        for name, value in gui_params.items():
            if hasattr(ace, name):
                getattr(ace, name).data.fill_(float(value))
        ace.band_gains.data.fill_(1.0)
    
    # Enable optimizations
    if enable_compile:
        ace.enable_compilation()
    
    return ace


def benchmark_comparison():
    """Compare original vs optimized ACE performance"""
    print("🏁 BENCHMARK: Original vs Fast ACE")
    print("=" * 40)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create test signal
    sample_rate = 44100
    duration = 2.0  # 2 seconds
    signal = torch.randn(1, 1, int(sample_rate * duration)) * 0.1
    signal = signal.to(device)
    
    print(f"Test signal: {duration}s on {device}")
    
    # Original ACE
    print("\\nTesting Original ACE...")
    original_ace = create_ace_with_gui_params(sample_rate=sample_rate).to(device)
    
    import time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    with torch.no_grad():
        orig_result = original_ace(signal)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    orig_time = time.time() - start
    
    # Fast ACE
    print("Testing Fast ACE...")
    fast_ace = create_fast_ace(sample_rate=sample_rate, training_mode=True).to(device)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    with torch.no_grad():
        fast_result = fast_ace(signal)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    fast_time = time.time() - start
    
    # Compare results
    print(f"\\n📊 Results:")
    print(f"Original ACE: {orig_time:.3f}s")
    print(f"Fast ACE:     {fast_time:.3f}s")
    print(f"Speedup:      {orig_time/fast_time:.2f}x")
    
    # Check output similarity
    output_diff = torch.mean((orig_result['output'] - fast_result['output'])**2).item()
    print(f"Output MSE:   {output_diff:.8f} (lower is better)")
    
    if output_diff < 1e-5:
        print("✅ Outputs are nearly identical!")
    else:
        print("⚠️  Outputs differ - check optimization settings")


if __name__ == "__main__":
    # Run benchmark
    benchmark_comparison()
    
    # Show usage example
    print("\\n" + "="*50)
    print("📖 USAGE EXAMPLES:")
    print("="*50)
    
    print("""
# Training-optimized ACE
fast_ace = create_fast_ace(
    sample_rate=44100,
    training_mode=True,      # Enable training optimizations
    chunk_size=8192,         # Process in chunks
    enable_compile=True      # Use torch.compile
)

# Use gradient checkpointing for large models
output = fast_ace(signal, use_checkpointing=True)

# For inference, disable training mode
fast_ace.training_mode = False
fast_ace.chunk_size = None  # Process full audio
""")


# Training tips
print("""
🚀 TRAINING OPTIMIZATION TIPS:
• Use chunk_size=4096-8192 for memory efficiency
• Enable gradient checkpointing for large batches  
• Use mixed precision training (AMP)
• Set training_mode=True during training
• Disable chunking during inference for best quality
""")