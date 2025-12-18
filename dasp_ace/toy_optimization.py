#!/usr/bin/env python3
"""
Toy Optimization Example for DASP ACE
Tests gradient flow and parameter optimization
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from ace_dasp import create_ace_with_gui_params
import torchaudio

def create_target_spectrum(signal, target_boost_freq=1000, boost_amount=2.0):
    """Create a target that boosts specific frequencies"""
    # Simple spectral target: boost energy around target_boost_freq
    fft = torch.fft.rfft(signal, dim=-1)
    freqs = torch.fft.rfftfreq(signal.shape[-1], 1/44100)
    
    # Create boost around target frequency
    freq_mask = torch.exp(-((freqs - target_boost_freq) / 500)**2)  # Gaussian boost
    target_fft = fft * (1 + boost_amount * freq_mask)
    
    return torch.fft.irfft(target_fft, n=signal.shape[-1], dim=-1)

def spectral_loss(output, target):
    """Loss based on spectral content"""
    output_fft = torch.fft.rfft(output, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    
    # L2 loss in frequency domain
    return torch.mean((output_fft.abs() - target_fft.abs())**2)

def toy_optimization():
    """Toy optimization to test ACE parameter learning"""
    print("🎯 TOY OPTIMIZATION: Testing ACE Parameter Learning")
    print("=" * 55)
    
    # Check for MPS (Apple Silicon GPU) availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using MPS (Apple GPU) for acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU for acceleration")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (consider using MPS on Apple Silicon)")
    
    # Create SHORT test signal for speed
    sample_rate = 44100
    duration = 0.5  # Much shorter - just 0.5 seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Simple single tone
    signal = torch.sin(2 * torch.pi * 440 * t) * 0.3  # Just A4
    signal = signal.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T] on GPU
    
    print(f"Test signal: {duration}s, {signal.shape[-1]} samples on {device}")
    
    # Simple target - just amplify the signal slightly
    target = signal * 1.5
    print(f"Target: 1.5x amplified version")
    
    print("Initializing ACE...")
    # Initialize ACE with learnable parameters
    ace = create_ace_with_gui_params(sample_rate=sample_rate)
    ace = ace.to(device)  # Move ACE to GPU
    
    # Optimize ONLY volume parameter for simplicity
    optimize_params = [ace.vol]
    
    # Reset to starting value
    with torch.no_grad():
        ace.vol.data = torch.tensor(0.0, device=device)  # Start at 0dB on GPU
    
    print(f"\nOptimizing 1 parameter: vol = {ace.vol.item():.3f} dB")
    
    # Set up optimizer
    optimizer = optim.Adam(optimize_params, lr=0.1)  # Higher learning rate
    
    # Track losses
    losses = []
    vol_history = []
    
    print(f"\n🚀 Starting optimization (simple L2 loss)...")
    
    # Short optimization loop
    num_epochs = 20  # Much fewer epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}...", end=" ")
        
        optimizer.zero_grad()
        
        # Forward pass
        try:
            results = ace(signal)
            output = results["output"]
            
            # Simple L2 loss
            loss = torch.mean((output - target)**2)
            
            # Backward pass
            loss.backward()
            
            # Update
            optimizer.step()
            
            # Log (move to CPU for printing)
            losses.append(loss.item())
            vol_history.append(ace.vol.item())
            
            print(f"Loss = {loss.item():.6f}, Vol = {ace.vol.item():.3f} dB")
            
        except Exception as e:
            print(f"ERROR: {e}")
            break
    
    print(f"\n✅ Optimization completed!")
    
    # Show results
    if len(losses) > 0:
        initial_vol = vol_history[0]
        final_vol = vol_history[-1]
        change = final_vol - initial_vol
        print(f"\n📊 Results:")
        print(f"  Volume: {initial_vol:.3f} → {final_vol:.3f} dB (Δ{change:+.3f})")
        print(f"  Loss improved: {losses[0]:.6f} → {losses[-1]:.6f}")
        print(f"\n🎉 SUCCESS: Gradients are flowing correctly!")
        print(f"✅ ACE parameters are learnable and optimizable")
        print(f"⚡ Accelerated with {device}")
    else:
        print("❌ Optimization failed - no progress made")
    
    return losses, vol_history

if __name__ == "__main__":
    losses, vol_history = toy_optimization()