# ACE Offline Processing Guide

This guide explains how to use the ACE (Auditory Contrast Enhancement) system to process WAV files offline.

## Prerequisites

1. **SuperCollider** must be installed
2. **ACEUGens** must be compiled and installed (see ACEUGens/README.md)
3. All ACE `.scd` files must be in the same directory

## Quick Start

### Option 1: Complete ACE Processing (Recommended)

1. **Start SuperCollider** and open `ace-offline-complete.scd`

2. **Load the script:**
   ```supercollider
   "path/to/ace-offline-complete.scd".load
   ```

3. **Process a file with default settings:**
   ```supercollider
   ~aceOfflineProcess.("input.wav", "output_enhanced.wav")
   ```

4. **Process with custom parameters:**
   ```supercollider
   ~aceOfflineProcess.("input.wav", "output_enhanced.wav", (
       wet: 0.8,        // 80% processed, 20% original
       vol: -3,         // -3dB output level
       rho: 25,         // Spectral contrast strength (0-30)
       beta: 6,         // Temporal contrast strength (0-10)
       dimWeight: 0.6   // 0=temporal only, 1=spectral only
   ))
   ```

### Option 2: Simplified Processing (If CGammatone UGen is not available)

1. **Load the simplified script:**
   ```supercollider
   "path/to/ace-offline.scd".load
   ```

2. **Process a file:**
   ```supercollider
   ~processACEOffline.("input.wav", "output_enhanced.wav")
   ```

## Parameters Explained

### Key ACE Parameters:

- **`wet`** (0.0-1.0): Dry/wet mix (0 = original, 1 = fully processed)
- **`vol`** (dB): Output volume adjustment
- **`rho`** (0-30): Spectral contrast enhancement strength
- **`beta`** (0-10): Temporal contrast enhancement strength  
- **`dimWeight`** (0.0-1.0): Balance between temporal (0) and spectral (1) enhancement

### Advanced Parameters:

- **`fHPF`** (Hz): High-pass filter frequency (default: 50 Hz)
- **`t60atCutoff`** (s): Decay time at cutoff frequency (default: 0.5s)
- **`tauLI`** (s): Lateral inhibition time constant (default: 7s)
- **`nu`** (dB): Noise floor for temporal processing (default: -40 dB)

## Example Usage

### Basic Enhancement:
```supercollider
// Load the system
"ace-offline-complete.scd".load

// Process with mild enhancement
~aceOfflineProcess.("speech.wav", "speech_enhanced.wav", (
    wet: 0.6,    // 60% processed
    vol: 0,      // No volume change
    rho: 15,     // Moderate spectral enhancement
    beta: 4     // Mild temporal enhancement
))
```

### Strong Enhancement for Difficult Audio:
```supercollider
~aceOfflineProcess.("noisy_audio.wav", "noisy_audio_enhanced.wav", (
    wet: 0.9,     // 90% processed
    vol: -3,      // Reduce level slightly
    rho: 25,      // Strong spectral enhancement
    beta: 8,      // Strong temporal enhancement
    dimWeight: 0.7 // Favor spectral processing
))
```

### Processing Multiple Files:
```supercollider
[
    "file1.wav",
    "file2.wav", 
    "file3.wav"
].do { |filename|
    var outputName = filename.replace(".wav", "_enhanced.wav");
    ~aceOfflineProcess.(filename, outputName, (wet: 0.8, vol: -3));
    "Processed: ".post; filename.postln;
}
```

## Troubleshooting

### Common Issues:

1. **"CGammatone UGen not available"**
   - The ACEUGens are not installed or not in SuperCollider's extension path
   - Use the simplified version instead, or compile ACEUGens following the instructions

2. **"ACE components not loaded"**
   - Make sure all `.scd` files are in the same directory
   - Check that the file paths are correct

3. **"Server not responding"**
   - Increase the processing time buffer in the script
   - Check SuperCollider server settings and memory allocation

4. **Poor audio quality****
   - Try different parameter combinations
   - Ensure input audio is good quality
   - Check for clipping with the `vol` parameter

### Performance Tips:

- For long files, ensure SuperCollider has enough memory allocated
- Process in smaller chunks if needed
- Use lower sample rates for faster processing (resample before/after)

## File Structure

After processing, you'll have:
```
input.wav           # Original file
input_enhanced.wav  # ACE-processed file
```

The enhanced file will have the same format and sample rate as the input.