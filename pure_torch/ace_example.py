#!/usr/bin/env python3
"""
Quick start example for ACE PyTorch implementation
Run this script to test the ACE processing on your audio files
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from ace_offline import process_audio_offline, ACETrainer, compare_parameters
    print("✓ ACE modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Main example function"""
    
    # Check if sample files exist
    wav_dir = current_dir / "wav"
    if not wav_dir.exists():
        wav_dir.mkdir()
        print(f"Created {wav_dir} directory")
        print("Please add some .wav files to the wav/ directory for testing")
        return
    
    # Find sample audio files
    audio_files = list(wav_dir.glob("*.wav"))
    if not audio_files:
        print("No .wav files found in wav/ directory")
        print("Please add some audio files for testing")
        return
    
    print(f"Found {len(audio_files)} audio files:")
    for f in audio_files[:3]:  # Show first 3
        print(f"  - {f.name}")
    
    # Process first audio file with default parameters
    input_file = audio_files[0]
    output_file = wav_dir / f"{input_file.stem}_ace_enhanced.wav"
    
    print(f"\n=== Processing {input_file.name} ===")
    
    try:
        # Default processing
        info = process_audio_offline(
            input_path=str(input_file),
            output_path=str(output_file)
        )
        
        print(f"✓ Enhanced audio saved to {output_file.name}")
        print(f"Duration: {info['duration_seconds']:.2f} seconds")
        
        # Show default parameters
        default_params = info['final_parameters']
        print("\nDefault ACE Parameters:")
        for param, value in default_params.items():
            print(f"  {param}: {value:.3f}")
        
        # Process with optimized parameters for better contrast
        optimized_output = wav_dir / f"{input_file.stem}_ace_optimized.wav"
        
        optimized_params = {
            'rho': 30.0,        # Higher spectral contrast  
            'beta': 12.0,       # Higher temporal contrast
            'wet': 0.85,        # More processed signal
            'dim_weight': 0.6,  # Balance spectral/temporal
            'vol': -3.0,        # Slightly quieter output
            'tau_li': 0.005,    # Faster lateral inhibition
            'tau_ex': 0.005,    # Faster expansion
        }
        
        print(f"\n=== Processing with optimized parameters ===")
        info_opt = process_audio_offline(
            input_path=str(input_file),
            output_path=str(optimized_output),
            ace_params=optimized_params
        )
        
        print(f"✓ Optimized audio saved to {optimized_output.name}")
        
        # Compare parameters
        compare_parameters(default_params, optimized_params)
        
        print(f"\n✓ Success! Compare these files:")
        print(f"  Original: {input_file.name}")  
        print(f"  Default ACE: {output_file.name}")
        print(f"  Optimized ACE: {optimized_output.name}")
        
    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()


def training_example():
    """Example of how to train ACE parameters (optional)"""
    print("\n=== Training Example (Optional) ===")
    
    wav_dir = Path("wav")
    audio_files = [str(f) for f in wav_dir.glob("*.wav")]
    
    if len(audio_files) < 2:
        print("Need at least 2 audio files for training example")
        return
    
    try:
        # Initialize trainer
        trainer = ACETrainer(sample_rate=44100)
        
        print("Training ACE model...")
        print("Note: This will take several minutes and requires GPU for speed")
        
        # Small training run
        trainer.train(
            audio_files=audio_files[:3],  # Use first 3 files
            epochs=5,                     # Short training
            batch_size=1,                 # Small batch for memory
            learning_rate=0.01,
            save_path="ace_trained_model"
        )
        
        print("✓ Training completed!")
        
        # Plot training history
        trainer.plot_training_history()
        
        # Test trained model
        test_file = audio_files[0]
        trained_output = wav_dir / f"{Path(test_file).stem}_ace_trained.wav"
        
        info_trained = process_audio_offline(
            input_path=test_file,
            output_path=str(trained_output),
            model_checkpoint="ace_trained_model_final.pt"
        )
        
        print(f"✓ Trained model output saved to {trained_output.name}")
        
    except Exception as e:
        print(f"✗ Training error: {e}")


if __name__ == "__main__":
    print("ACE PyTorch - Quick Start Example")
    print("=" * 40)
    
    main()
        # Uncomment the line below to run training example
    # training_example()
    
    print("\n" + "=" * 40)
    print("Next steps:")
    print("1. Listen to the enhanced audio files")  
    print("2. Adjust parameters in optimized_params dict")
    print("3. Use ACETrainer class for ML optimization")
    print("4. Integrate with your audio processing pipeline")