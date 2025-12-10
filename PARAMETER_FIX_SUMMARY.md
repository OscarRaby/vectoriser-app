# Parameter Binding & Segmentation Fixes - December 10, 2025

## Problem Summary

All parameters were appearing at **zero** in the UI and in pipeline execution, resulting in:
1. Single-color output images (no segmentation occurring)
2. Same output regardless of slider position
3. Pipeline logs showing `noiseScale=1E-06` (minimum fallback value)

## Root Causes Identified

### 1. **WPF Binding Initialization Issue**
**Problem**: Two-way bindings in WPF don't always pull source values during initial DataContext setup.
- Slider default value (0) would take precedence
- Binding would push that 0 back to the ParameterSet property
- This overwrote the default values (NoiseScale=60, BlurSigma=2, etc.)

**Solution**: Added Window.Loaded event in MainWindow.xaml.cs to trigger PropertyChanged notifications, forcing UI bindings to refresh from source values.

### 2. **Incorrect Slider Range**
**Problem**: XAML Slider had range 0-1000, but Python version uses 10-200.
- Range didn't match the intended parameter domain
- Default value of 60 was only 6% into the 0-1000 range

**Solution**: Changed Slider range in MainWindow.xaml to Minimum="10" Maximum="200" to match Python version.

### 3. **Inverted Parameter Scaling Formula** (CRITICAL)
**Problem**: The most critical issue - noise scale was being multiplied by image scale factor instead of divided.

**Before (Wrong)**:
```csharp
var scaleFactor = GetScaleFactor(input.Width, input.Height);  // 5.33 for 800x600
var noiseScale = Math.Max(1e-6, p.NoiseScale * scaleFactor / multiplier);
// Result: 60 * 5.33 / 1 = 320
// Grid: 800 / 320 = 2.5 → only 2-3 cells across width!
```

**After (Correct)**:
```csharp
var noiseScale = Math.Max(1.0, p.NoiseScale * multiplier / scaleFactor);
// Result: 60 * 1 / 5.33 = 11.25
// Grid: 800 / 11.25 = 71 cells across width!
```

This matches Python's pnoise2 behavior where scale represents frequency domain sampling.

## Changes Made

### File: `MainWindow.xaml.cs`
```csharp
this.Loaded += (s, e) =>
{
    if (DataContext is MainViewModel vm)
    {
        // Trigger PropertyChanged to refresh bindings
        vm.Parameters.NoiseScale = vm.Parameters.NoiseScale;
        vm.Parameters.BlurSigma = vm.Parameters.BlurSigma;
        // ... etc for all key parameters
    }
};
```

### File: `MainWindow.xaml`
```xml
<!-- Before -->
<Slider Minimum="0" Maximum="1000" TickFrequency="10" 
    Value="{Binding Parameters.NoiseScale, Mode=TwoWay}" />

<!-- After -->
<Slider Minimum="10" Maximum="200" TickFrequency="5" 
    Value="{Binding Parameters.NoiseScale, Mode=TwoWay}" />
```

### File: `PipelineService.cs` (Lines 62-69)
```csharp
// Changed parameter scaling for noise from: p.NoiseScale * scaleFactor / multiplier
// To: p.NoiseScale * multiplier / scaleFactor (INVERTED)
var noiseScale = Math.Max(1.0, p.NoiseScale * multiplier / scaleFactor);
var blurSigma = Math.Max(1e-6, p.BlurSigma * scaleFactor / multiplier);
var compactness = Math.Max(1e-6, p.Compactness * scaleFactor / multiplier);
```

## Expected Results After Fix

1. **Slider shows correct values**: Sliders will display non-zero values (not stuck at 0)
2. **Parameter changes affect output**: Moving NoiseScale slider from 10 to 200 should produce visibly different segmentation
3. **Proper segmentation grain**: Image should segmentinto multiple regions (not a single color)
4. **Scale consistency**: Different sized images will have consistent segmentation grain size

## Testing Instructions

1. **Launch the application**
2. **Load an image** (File → Load Image)
3. **Verify slider positions**:
   - NoiseScale slider should show ~60 (middle position with 10-200 range)
   - BlurSigma should show ~2
   - Other parameters should show their defaults
4. **Test parameter sensitivity**:
   - Click Preview with NoiseScale at 10 (left)
   - Note the output image
   - Move NoiseScale to 200 (right)
   - Click Preview again
   - Output should be noticeably different (coarser/finer segmentation)
5. **Test different images**:
   - Load different images
   - Verify consistent segmentation behavior regardless of image size

## Expected Parameter Behavior

With these fixes:
- **Noise Scale 10**: Very fine segmentation (many small regions)
- **Noise Scale 60**: Medium segmentation (good default balance)
- **Noise Scale 200**: Coarse segmentation (fewer, larger regions)

Each slider position should now produce distinctly different results.

## Technical Explanation

### Why the scaling formula matters:

The Perlin noise generator needs to create a grid of gradient vectors. The scale parameter determines the spacing between these vectors in frequency space:

- **Small scale** (e.g., 11.25): Creates frequent oscillations, fine grain
- **Large scale** (e.g., 180): Creates sparse oscillations, coarse grain

For an 800-pixel-wide image:
- With scale=11.25: Creates 71 gradient points across the width (very detailed)
- With scale=180: Creates 4-5 gradient points across the width (very coarse)

The fix ensures that when users slide the NoiseScale parameter, they're actually changing this behavior as intended.

## Build Status

✅ **Compilation**: 0 errors, 2 warnings (unrelated)
✅ **Runtime**: Application launches and runs successfully
✅ **Pipeline**: All stages execute without crashes

---
*All fixes have been applied and tested to compile successfully.*
