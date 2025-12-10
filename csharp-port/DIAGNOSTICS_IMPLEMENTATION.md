# Pipeline Diagnostics Tool - Implementation Summary

## What Was Added

A complete diagnostic system to isolate and test each pipeline stage independently.

## Files Created/Modified

### New Files:
1. **DiagnosticsService.cs** - Core diagnostic engine with 4 test methods:
   - `TestNoiseGeneration()` - Isolate Perlin noise generation
   - `TestElevationMap()` - Test elevation map creation  
   - `TestLocalMinima()` - Test marker point detection
   - `TestWatershed()` - Test watershed segmentation

2. **DIAGNOSTICS_GUIDE.md** - Detailed usage and interpretation guide
3. **QUICK_DEBUG_STEPS.md** - Quick checklist for debugging

### Modified Files:
1. **PipelineService.cs**
   - Added `RunDiagnosticsAsync()` method to interface
   - Implemented full diagnostic pipeline
   - Added detailed console logging

2. **MainViewModel.cs**
   - Added `DiagnosticsCommand` (AsyncRelayCommand)
   - Added `RunDiagnosticsAsync()` method
   - Shows results dialog with path to diagnostic images

3. **MainWindow.xaml**
   - Added yellow "Run Diagnostics" button
   - Placed after Run Vectoriser button for easy access

## How It Works

### User Flow:
```
Load Image → Click "Run Diagnostics" 
→ App processes through 4 pipeline stages
→ Generates 4 diagnostic images
→ Prints detailed console output
→ Shows results dialog
```

### Output Locations:
- **Diagnostic Images**: `%APPDATA%\OrganicVectoriser\diagnostics\`
  - 01_noise_scale[N].png - Raw noise field
  - 02_elevation_blur[N]_noise[M].png - Elevation map
  - 03_local_maxima.png - Marker points (watershed seeds)
  - 04_watershed_result.png - Final segmentation

- **Console Output**: Debug terminal shows:
  - Parameter values and scale factors
  - Numeric ranges for each stage
  - Count of markers and segments found
  - Label statistics (how many pixels per region)

## What This Reveals

### Stage 1: Noise Generation
Tests if Perlin noise is being created correctly
- **Good**: Shows gradual light/dark variation
- **Bad**: Completely uniform (single gray color)

### Stage 2: Elevation Map
Tests the combined grayscale + noise elevation map
- **Good**: Shows texture variation and structure
- **Bad**: Smooth without variation

### Stage 3: Local Minima
Tests detection of potential watershed marker points
- **Good**: Multiple white dots across image
- **Bad**: Mostly black (no markers detected)

### Stage 4: Watershed
Tests the actual watershed segmentation algorithm
- **Good**: Multiple colors (each is a segment)
- **Bad**: One dominant color (all pixels in one segment)

## The Problem Your Code Found

You observed: **"Single color output, different color each time"**

This suggests:
- ✅ Randomness works (random color each time)
- ✅ Watershed is running (at least creating one segment)
- ❌ Watershed is creating only 1 label (not segmenting)

### Likely Causes:
1. **Elevation map has no structure** - No local variation
2. **No marker points detected** - Watershed has nothing to seed with
3. **Parameters causing complete smoothing** - Blur too high, noise too low

## Next Steps

1. **Run diagnostics on your test image**
2. **Check each diagnostic image in order**
3. **Identify which stage first shows the problem**
4. **Report findings** and we'll fix that specific stage

## Example Diagnostic Output

```
[DIAGNOSTICS] Image: 800x600
[DIAGNOSTICS] Scale factor: 5.33
[DIAGNOSTICS] Scaled NoiseScale: 11.25
[DIAGNOSTICS] Scaled BlurSigma: 1.88

[DIAG] Noise value range: [-0.345, 0.420] ← Good variation
[DIAG] Elevation range: [0.120, 0.890] ← Good range
[DIAG] Found 127 potential marker points ← Good count
[DIAG] Connected components found: 128 ← Good markers
[DIAG] Watershed output unique labels: 215 ← Good segmentation!

Label Statistics:
  Total unique labels: 215
  Label 0: 45% (main segment 1)
  Label 1: 35% (main segment 2)
  Label 5: 12% (sub-segment)
  ... 212 more regions
```

If you're only getting 1-2 labels, the diagnostics will pinpoint which stage is failing.

---

**Next: Run diagnostics and share the output so we can identify the exact problem!**
