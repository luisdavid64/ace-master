"""
ACE Offline Processing and Training Script
Provides utilities for processing audio files and optimizing ACE parameters
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import matplotlib.pyplot as plt
from ace_pytorch import ACEProcessor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    """Dataset for loading audio files for ACE training"""
    
    def __init__(self, 
                 audio_files: List[Union[str, Path]], 
                 sample_rate: int = 44100,
                 chunk_duration: float = 2.0,
                 overlap: float = 0.5):
        self.audio_files = [Path(f) for f in audio_files]
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap * self.chunk_samples)
        self.stride = self.chunk_samples - self.overlap_samples
        
        # Preload and chunk audio files
        self.chunks = []
        self._load_audio_chunks()
    
    def _load_audio_chunks(self):
        """Load audio files and split into chunks"""
        for audio_file in self.audio_files:
            try:
                waveform, sr = torchaudio.load(audio_file)
                
                # Resample if necessary
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Split into chunks
                for start in range(0, waveform.shape[1] - self.chunk_samples, self.stride):
                    end = start + self.chunk_samples
                    chunk = waveform[:, start:end]
                    
                    # Ensure chunk is exactly the right length
                    if chunk.shape[1] == self.chunk_samples:
                        # Convert to stereo for ACE processing
                        if chunk.shape[0] == 1:
                            chunk = chunk.repeat(2, 1)
                        self.chunks.append(chunk)
                        
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]


class ACETrainer:
    """Training utility for ACE parameter optimization"""
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.sample_rate = sample_rate
        self.device = device
        
        # Initialize ACE model
        self.ace_model = ACEProcessor(sample_rate=sample_rate).to(device)
        
        # Training history
        self.history = {
            'loss': [],
            'parameters': []
        }
    
    def spectral_centroid_loss(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Loss function based on spectral centroid enhancement
        Encourages the enhanced audio to have better spectral clarity
        """
        # Compute spectral centroids
        def compute_spectral_centroid(x):
            # Simple FFT-based spectral centroid
            X = torch.fft.rfft(x, dim=-1)
            magnitude = torch.abs(X)
            
            # Frequency bins
            freqs = torch.linspace(0, self.sample_rate//2, magnitude.shape[-1], device=x.device)
            freqs = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, freq_bins]
            
            # Weighted average frequency
            centroid = torch.sum(magnitude * freqs, dim=-1) / (torch.sum(magnitude, dim=-1) + 1e-8)
            return centroid.mean()
        
        original_centroid = compute_spectral_centroid(original)
        enhanced_centroid = compute_spectral_centroid(enhanced)
        
        # Encourage higher spectral centroid (brighter sound)
        centroid_loss = -torch.log(enhanced_centroid / (original_centroid + 1e-3) + 1e-8)
        
        return centroid_loss
    
    def dynamic_range_loss(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Loss function to encourage better dynamic range
        """
        # Compute RMS in overlapping windows
        window_size = 1024
        hop_size = 512
        
        def compute_rms_windows(x):
            rms_values = []
            for i in range(0, x.shape[-1] - window_size, hop_size):
                window = x[..., i:i+window_size]
                rms = torch.sqrt(torch.mean(window**2, dim=-1))
                rms_values.append(rms)
            return torch.stack(rms_values, dim=-1)
        
        original_rms = compute_rms_windows(original)
        enhanced_rms = compute_rms_windows(enhanced)
        
        # Compute standard deviation of RMS (dynamic range indicator)
        original_dynamics = torch.std(original_rms, dim=-1)
        enhanced_dynamics = torch.std(enhanced_rms, dim=-1)
        
        # Encourage better dynamics
        dynamics_loss = -torch.log(enhanced_dynamics / (original_dynamics + 1e-3) + 1e-8)
        
        return dynamics_loss.mean()
    
    def perceptual_loss(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Combined perceptual loss function
        """
        # Spectral contrast enhancement
        spectral_loss = self.spectral_centroid_loss(original, enhanced)
        
        # Dynamic range enhancement
        dynamics_loss = self.dynamic_range_loss(original, enhanced)
        
        # Preserve naturalness (don't deviate too much from original)
        mse_loss = torch.mean((enhanced - original)**2)
        naturalness_loss = torch.sqrt(mse_loss)
        
        # Combined loss with weights
        total_loss = (0.3 * spectral_loss + 
                     0.3 * dynamics_loss + 
                     0.4 * naturalness_loss)
        
        return total_loss
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Train for one epoch"""
        self.ace_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, audio_batch in enumerate(dataloader):
            audio_batch = audio_batch.to(self.device)
            batch_size = audio_batch.shape[0]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            enhanced_audio = self.ace_model(audio_batch)
            
            # Compute loss
            loss = self.perceptual_loss(audio_batch, enhanced_audio)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.ace_model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def train(self, 
              audio_files: List[str], 
              epochs: int = 50,
              batch_size: int = 4,
              learning_rate: float = 0.001,
              save_path: Optional[str] = None):
        """
        Train ACE model on audio files
        """
        # Create dataset and dataloader
        dataset = AudioDataset(audio_files, sample_rate=self.sample_rate)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = optim.Adam(self.ace_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        print(f"Training on {len(dataset)} audio chunks...")
        print(f"Initial parameters: {self.ace_model.get_parameter_dict()}")
        
        for epoch in range(epochs):
            # Train epoch
            avg_loss = self.train_epoch(dataloader, optimizer)
            
            # Update scheduler
            scheduler.step(avg_loss)
            
            # Store history
            self.history['loss'].append(avg_loss)
            self.history['parameters'].append(self.ace_model.get_parameter_dict().copy())
            
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if save_path and (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"{save_path}_epoch_{epoch+1}.pt")
        
        print("Training completed!")
        print(f"Final parameters: {self.ace_model.get_parameter_dict()}")
        
        if save_path:
            self.save_checkpoint(f"{save_path}_final.pt")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.ace_model.state_dict(),
            'parameters': self.ace_model.get_parameter_dict(),
            'history': self.history
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.ace_model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {'loss': [], 'parameters': []})
        print(f"Checkpoint loaded from {path}")
        return checkpoint['parameters']
    
    def plot_training_history(self):
        """Plot training loss and parameter evolution"""
        if not self.history['loss']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Key parameter evolution
        if self.history['parameters']:
            epochs = range(len(self.history['parameters']))
            
            # Plot rho (spectral contrast)
            rho_values = [p['rho'] for p in self.history['parameters']]
            axes[0, 1].plot(epochs, rho_values)
            axes[0, 1].set_title('Rho (Spectral Contrast)')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].grid(True)
            
            # Plot beta (temporal contrast)
            beta_values = [p['beta'] for p in self.history['parameters']]
            axes[1, 0].plot(epochs, beta_values)
            axes[1, 0].set_title('Beta (Temporal Contrast)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].grid(True)
            
            # Plot wet mix
            wet_values = [p['wet'] for p in self.history['parameters']]
            axes[1, 1].plot(epochs, wet_values)
            axes[1, 1].set_title('Wet Mix')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('ace_training_history.png', dpi=300)
        plt.show()


def process_audio_offline(input_path: str, 
                         output_path: str, 
                         ace_params: Optional[Dict] = None,
                         model_checkpoint: Optional[str] = None,
                         sample_rate: int = 44100) -> Dict:
    """
    Process audio file with ACE offline
    
    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        ace_params: Dictionary of ACE parameters to override
        model_checkpoint: Path to trained model checkpoint
        sample_rate: Audio sample rate
    
    Returns:
        Dictionary with processing info and final parameters
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize ACE model
    ace_model = ACEProcessor(sample_rate=sample_rate).to(device)
    
    # Load checkpoint if provided
    if model_checkpoint:
        checkpoint = torch.load(model_checkpoint, map_location=device)
        ace_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_checkpoint}")
    
    # Override parameters if provided
    if ace_params:
        for param_name, value in ace_params.items():
            if hasattr(ace_model, param_name):
                getattr(ace_model, param_name).data.fill_(value)
                print(f"Set {param_name} = {value}")
    
    # Load input audio
    print(f"Loading {input_path}...")
    waveform, sr = torchaudio.load(input_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        print(f"Resampled from {sr} Hz to {sample_rate} Hz")
    
    # Ensure stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]  # Take first two channels
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0).to(device)
    
    print(f"Processing audio: {waveform.shape[2]/sample_rate:.2f} seconds...")
    
    # Process with ACE
    ace_model.eval()
    with torch.no_grad():
        enhanced_waveform = ace_model(waveform)
    
    # Remove batch dimension and move to CPU
    enhanced_waveform = enhanced_waveform.squeeze(0).cpu()
    
    # Save output
    torchaudio.save(output_path, enhanced_waveform, sample_rate)
    print(f"Saved enhanced audio to {output_path}")
    
    # Return processing info
    return {
        'input_path': input_path,
        'output_path': output_path,
        'sample_rate': sample_rate,
        'duration_seconds': waveform.shape[2] / sample_rate,
        'final_parameters': ace_model.get_parameter_dict()
    }


def compare_parameters(original_params: Dict, optimized_params: Dict) -> None:
    """Compare original vs optimized ACE parameters"""
    print("\nParameter Comparison:")
    print("-" * 50)
    print(f"{'Parameter':<15} {'Original':<12} {'Optimized':<12} {'Change':<10}")
    print("-" * 50)
    
    for param_name in original_params.keys():
        if param_name in optimized_params:
            orig_val = original_params[param_name]
            opt_val = optimized_params[param_name]
            change = opt_val - orig_val
            print(f"{param_name:<15} {orig_val:<12.3f} {opt_val:<12.3f} {change:+.3f}")


if __name__ == "__main__":
    # Example usage
    
    # 1. Process single file with default parameters
    print("=== Processing with default parameters ===")
    info = process_audio_offline(
        input_path="wav/sampleA.wav",
        output_path="wav/sampleA_ace_default.wav"
    )
    
    # 2. Process with custom parameters
    print("\n=== Processing with custom parameters ===")
    custom_params = {
        'rho': 30.0,      # Higher spectral contrast
        'beta': 10.0,     # Higher temporal contrast  
        'wet': 0.8,       # 80% processed signal
        'dim_weight': 0.7 # More spectral emphasis
    }
    
    info_custom = process_audio_offline(
        input_path="wav/sampleA.wav",
        output_path="wav/sampleA_ace_custom.wav",
        ace_params=custom_params
    )
    
    # 3. Train model on audio files (example)
    print("\n=== Training example (uncomment to run) ===")
    """
    trainer = ACETrainer()
    
    # Collect audio files for training
    audio_files = [
        "wav/sampleA.wav",
        "wav/sampleB.wav"
        # Add more training files here
    ]
    
    # Train model
    trainer.train(
        audio_files=audio_files,
        epochs=20,
        batch_size=2,
        learning_rate=0.001,
        save_path="ace_trained_model"
    )
    
    # Plot training progress
    trainer.plot_training_history()
    
    # Process with trained model
    process_audio_offline(
        input_path="wav/sampleA.wav",
        output_path="wav/sampleA_ace_trained.wav",
        model_checkpoint="ace_trained_model_final.pt"
    )
    """
    
    print("\nACE PyTorch implementation ready for optimization!")