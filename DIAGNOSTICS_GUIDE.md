# Pipeline Diagnostics Guide

## Overview

A comprehensive diagnostic tool has been added to isolate and test each stage of the pipeline independently. This will help identify exactly where the segmentation is failing.

## How to Use

### 1. **Launch the App**
The app now has a new button: **"Run Diagnostics"** (yellow button) in the middle panel.

### 2. **Load an Image**
- Click "Load Image"
- Select any image (PNG, JPG, etc.)

### 3. **Run Diagnostics**
- Click the yellow "Run Diagnostics" button
- The app will:
  - Process through each pipeline stage
  - Generate diagnostic images for each stage
  - Print detailed debug information to the console
  - Save all intermediate results to disk

### 4. **View Results**
A dialog will appear showing where the diagnostic images were saved:
```
C:\Users\[YourName]\AppData\Roaming\OrganicVectoriser\diagnostics\
```

## What Each Diagnostic Image Shows

### 1. **01_noise_scale[N].png**
**What**: The Perlin noise field being used for segmentation
**Why it matters**: 
- Should show variation across the image (not uniform)
- Darker/lighter regions indicate frequency variation
- **If uniform**: The noise generation isn't working correctly
- **If varies**: Noise is fine, problem is downstream

### 2. **02_elevation_blur[N]_noise[M].png**
**What**: The combined elevation map (0.7 × blurred + 0.3 × noise)
**Why it matters**:
- Should show the blended grayscale image with noise texture overlay
- Lighter areas = potential watershed divide points
- Darker areas = potential region centers
- **If uniform/smooth**: Watershed has no variation to work with

### 3. **03_local_maxima.png**
**What**: The detected local maximum points that become watershed markers
**Why it matters**:
- White pixels = marker points
- Should see multiple white dots/clusters across the image
- **If mostly black**: No markers detected → this is the problem!
- **If sparse white dots**: May be too few markers

### 4. **04_watershed_result.png**
**What**: The final watershed segmentation result (colored)
**Why it matters**:
- Different colors = different segments/regions
- **If mostly one color**: Watershed failed (all pixels same label)
- **If many colors**: Segmentation worked!

## Console Output

The app will also print detailed logs to the console showing:

```
[DIAGNOSTICS] Starting diagnostic pipeline
[DIAGNOSTICS] Image: 800x600
[DIAGNOSTICS] Scale factor: 5.33
[DIAGNOSTICS] Scaled NoiseScale: 11.25
[DIAGNOSTICS] Scaled BlurSigma: 1.88

=== STAGE 1: NOISE GENERATION ===
[DIAG] Testing noise generation: 800x600, scale=11.25
[DIAG] Noise value range: [-0.3450, 0.4200]

=== STAGE 2: ELEVATION MAP ===
[DIAG] Testing elevation map: blur_sigma=1.88, noise_scale=11.25
[DIAG] Gray: min=0.0000, max=1.0000
[DIAG] Blurred with ksize=13
[DIAG] Elevation range: [0.1200, 0.8900]

=== STAGE 3: LOCAL MINIMA ===
[DIAG] Testing local minima detection on elevation map
[DIAG] Distance transform range: [0.00, 425.32]
[DIAG] Local maxima threshold: 127.60
[DIAG] Found 127 potential marker points

=== STAGE 4: WATERSHED ===
[DIAG] Testing watershed segmentation
[DIAG] Connected components found: 128
[DIAG] Watershed output unique labels: 215
[DIAG] Trying to understand why it's single color?

=== STATISTICS ===
Label Statistics for 600x800 image:
  Total unique labels: 2
  Label 0: 479990 pixels (99.99%)
  Label -1: 10 pixels (0.01%)
```

## Interpreting the Results

### ✅ **Good Signs**:
- Noise value range is NOT all zeros (e.g., `[-0.3450, 0.4200]`)
- Found reasonable number of markers (50+)
- Watershed output has many unique labels (100+)
- Label statistics show multiple labels with reasonable percentages

### 🔴 **Problem Signs**:
- Noise range shows `[0, 0]` → Noise generation broken
- Found 0 marker points → Elevation map has no structure
- Watershed shows only 1-2 unique labels → Segmentation failed
- Single label with 99%+ of pixels → Complete segmentation failure

## Common Issues & Fixes

### Issue: Only 1-2 Labels in Output
**Possible Cause**: 
- Parameters causing division by zero or near-zero values
- Elevation map too smooth (no local minima)

**Check**:
1. Is `Scaled NoiseScale` reasonable? (Should be 5-50, not 0 or 1000+)
2. Is `Scaled BlurSigma` reasonable? (Should be 0.5-5, not 0 or huge)
3. Are there marker points detected? (Should be 50+)

### Issue: No Marker Points Detected
**Possible Cause**:
- Elevation map is too uniform
- Threshold for local maxima too strict

**Check**:
1. Look at `02_elevation_*.png` - does it have texture variation?
2. Check distance transform range - is max value large enough?

### Issue: Noise Not Varying
**Possible Cause**:
- Noise scale parameter wrong
- Perlin noise generator broken

**Check**:
1. Is `Noise value range` showing variation? 
2. Look at `01_noise_scale*.png` - should see gradual variation, not uniform

## Next Steps

**If diagnostics show multiple segments (✓ good)**:
- Problem is in post-processing stages (quantization, contour extraction, etc.)
- We need to debug those stages

**If diagnostics show single segment (✗ bad)**:
- Problem is in watershed or upstream (noise/elevation)
- Check the specific diagnostic images and console output
- May need to adjust scaling formula again

## Manual Testing

You can also manually test individual components:

```csharp
// Pseudocode for manual testing
var diag = new DiagnosticsService();
var noise = diag.TestNoiseGeneration(800, 600, 11.25);
var elevation = diag.TestElevationMap(rgb, 1.88, 11.25);
var markers = diag.TestLocalMinima(elevation);
var result = diag.TestWatershed(rgb, elevation);
```

---

**Run the diagnostics now and share the console output and the diagnostic image folder contents so we can see exactly where the pipeline is failing!**
