# Quick Diagnostics Checklist

**To isolate the problem, follow these steps:**

## Step 1: Run Diagnostics
1. Load an image in the app
2. Click the yellow **"Run Diagnostics"** button
3. Wait for the message showing where images were saved

## Step 2: Check Console Output
Look for these values in the console:

```
[DIAGNOSTICS] Scaled NoiseScale: X.XX
[DIAGNOSTICS] Found N potential marker points
[DIAG] Watershed output unique labels: N
```

### What Should You See?
- ✅ Scaled NoiseScale: 5-50 (NOT 0, NOT 1000+)
- ✅ Found marker points: 50+ (NOT 0, NOT 1)
- ✅ Unique labels: 50+ (NOT 1-2)

## Step 3: Check Diagnostic Images

Open the diagnostics folder:
```
C:\Users\[YourName]\AppData\Roaming\OrganicVectoriser\diagnostics\
```

### Image 01: Noise
- Should show gradual variation (light/dark regions)
- If completely uniform (one solid gray): **NOISE GENERATION FAILED**

### Image 02: Elevation Map
- Should show the grayscale image with texture overlay
- If completely smooth: **ELEVATION MAP HAS NO STRUCTURE**

### Image 03: Local Maxima
- Should show white dots/clusters scattered across the image
- If mostly black with few/no white dots: **NO MARKERS DETECTED**

### Image 04: Watershed Result
- Should show many different colors (each = one segment)
- If mostly one color: **WATERSHED FAILED**

## Step 4: Report Findings

Based on which diagnostic image shows the problem, you'll know:

| Image | If Failed | Problem Location |
|-------|-----------|------------------|
| 01_noise | Uniform | Perlin noise generation |
| 02_elevation | Smooth | Blur/noise combination |
| 03_maxima | Mostly black | Local minima detection |
| 04_watershed | One color | Watershed segmentation |

---

**Ready to test? Load an image and click "Run Diagnostics"!**
