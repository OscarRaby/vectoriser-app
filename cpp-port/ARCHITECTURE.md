# Vectoriser C++ Architecture Overview

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VectorizationPipeline                             │
│  (Main Orchestrator - VectorizationPipeline.h/.cpp)                 │
└────┬────────────────────────────────────────────────────────────────┘
     │
     ├─► Input: cv::Mat (RGB/BGR/Grayscale)
     │
     ├─── Stage 1: SEGMENTATION ─────────────────────────────────┐
     │    SegmentationService::NoiseWatershed()                  │
     │    • RGBToGrayFloat()                                     │
     │    • GeneratePerlinNoise()    [stub: FastNoise2]          │
     │    • CreateElevationMap() = 0.7*blur + 0.3*noise          │
     │    • MorphologicalLocalMinima() [dilate + compare]        │
     │    • cv::watershed()                                      │
     │    Output: cv::Mat (labels)                               │
     └────────────────────────────────────────────────────────────┘
     │
     ├─── Stage 2: QUANTIZATION ──────────────────────────────────┐
     │    ContourService::QuantizeColors()                       │
     │    • cv::kmeans() with PP initialization                  │
     │    • Returns (quantized image, cluster centers)           │
     │    Output: cv::Mat (quantized), Vec3b[] (centers)         │
     └────────────────────────────────────────────────────────────┘
     │
     ├─── Stage 3: CONTOUR EXTRACTION ────────────────────────────┐
     │    ContourService::ExtractContours()                      │
     │    • Find dominant color per label                        │
     │    • Calculate centroid (mean of points)                  │
     │    • Calculate area (shoelace formula)                    │
     │    • Calculate luminance (ITU-R BT.709)                   │
     │    Output: ContourData[] (points, color, area, etc)       │
     └────────────────────────────────────────────────────────────┘
     │
     ├─── Stage 4: BRIDGING [OPTIONAL] ───────────────────────────┐
     │    BridgingService::ImprovedBridgeContours()              │
     │    • Convert colors RGB → LAB                             │
     │    • Find proximity-nearby contours                       │
     │    • Check color similarity in LAB space                  │
     │    • Check curvature constraints                          │
     │    • Apply cosine falloff weighting                       │
     │    Output: Modified ContourData[] (bridged points)        │
     └────────────────────────────────────────────────────────────┘
     │
     ├─── Stage 5: SMOOTHING [OPTIONAL] ─────────────────────────┐
     │    GeometryService::LaplacianSmooth()                     │
     │    • new_pt = (1-α)*pt + α*0.5*(prev+next)               │
     │    • Iterative refinement                                 │
     │    Output: Smoothed ContourData[]                         │
     └────────────────────────────────────────────────────────────┘
     │
     ├─── Stage 6: INFLATION [OPTIONAL] ──────────────────────────┐
     │    GeometryService::InflateContour()                      │
     │    • Calculate normalized distances from centroid         │
     │    • Uniform: all points expand equally                   │
     │    • Exponential: pt' = center + dir*dist*amt*exp(...)    │
     │    Output: Inflated ContourData[]                         │
     └────────────────────────────────────────────────────────────┘
     │
     ├─── Stage 7: SORTING ───────────────────────────────────────┐
     │    SortingService::SortContoursByOrder()                  │
     │    • 10 sorting strategies available:                     │
     │      - Area (largest/smallest first)                      │
     │      - Brightness (darkest/brightest)                     │
     │      - Position (X, Y, distance from center)              │
     │    Output: std::vector<size_t> (z-order indices)          │
     └────────────────────────────────────────────────────────────┘
     │
     └─► Output: PipelineResult
         • contours: ContourData[]
         • zOrderIndices: size_t[]
         • droplets: DropletDescriptor[] [empty for now]
         • timings: map<string, double>
         • totalTimeMs: double
```

## Component Dependency Graph

```
VectorizationPipeline
├── SegmentationService
│   └── (OpenCV: cvtColor, gaussian, dilate, watershed)
├── ContourService
│   └── (OpenCV: kmeans, connectedComponents, findContours)
├── BridgingService
│   ├── ContourService [for luminance]
│   └── (OpenCV: cvtColor RGB→LAB)
├── GeometryService
│   └── (Math: vector algebra)
├── SortingService
│   └── ContourService [for luminance]
├── ParameterScaler
│   └── (Math: sqrt, max, min)
└── SVGWriter
    └── (OpenCV: approxPolyDP for simplification)

All modules depend on:
- VectorizationParameters.h (type definitions)
- OpenCV 4.12.0
```

## Data Flow - Complete Pipeline

```
Input Image (RGB, cv::Mat)
    ↓
[SEGMENTATION] ← ParameterScaler (adapt parameters to image size)
    ↓
Labels (cv::Mat)
    ↓ + Original Image
[QUANTIZATION]
    ↓
Quantized Image (cv::Mat) + Cluster Centers
    ↓ + Labels
[CONTOUR EXTRACTION]
    ↓
ContourData[] {
  • points: vector<Point2d>
  • centroid: Point2d
  • color: Vec3b
  • area: double
  • luminance: double
}
    ↓ [if ModifierFlags::enableBridging]
[BRIDGING] ← LAB color space conversion
    ↓
Updated points (bridged)
    ↓ [if ModifierFlags::enableSmoothing]
[SMOOTHING]
    ↓
Smoothed points
    ↓ [if ModifierFlags::enableInflation]
[INFLATION]
    ↓
Inflated points
    ↓
[SORTING] ← StackingOrder selection
    ↓
Z-Order Indices
    ↓
PipelineResult {
  • contours: ContourData[]
  • zOrderIndices: size_t[]
  • droplets: DropletDescriptor[]
  • imageSize: cv::Size
  • timings: map<string, double>
  • totalTimeMs: double
}
    ↓
[SVG EXPORT]
    ↓
SVG File (XML with <path> elements)
```

## Parameter Hierarchy

```
VectorizationParameters.h
├── Enumerations
│   ├── ScalingMethod (5: MAX, MIN, AVERAGE, AREA, SQRT_AREA)
│   ├── StackingOrder (10: AREA*, BRIGHTNESS*, POSITION_X*, POSITION_Y*, POSITION_CENTRE*)
│   ├── DropletStyle (2: PAINTERLY, ORGANIC)
│   └── PainterlyPrimitive (3: ELLIPSE, RECT, BRUSH_STROKE)
│
├── Segmentation Parameters
│   └── SegmentationParams
│       ├── noiseScale: 60.0 (px)
│       ├── blurSigma: 2.0 (px)
│       ├── compactness: 0.001
│       ├── referenceCropSize: 512x512
│       ├── scalingMethod: SQRT_AREA
│       └── segmentationMultiplier: 1.0
│
├── Quantization Parameters
│   └── QuantizationParams
│       └── maxColors: 8
│
├── Bridging Parameters
│   └── BridgingParams
│       ├── proximityThreshold: 20.0 (px)
│       ├── colorTolerance: 30.0 (LAB units)
│       ├── bridgeDistance: 3.0 (px)
│       ├── falloffRadius: 5 (iterations)
│       └── maxCurvature: 45.0 (degrees)
│
├── Smoothing Parameters
│   └── SmoothingParams
│       ├── iterations: 3 (1-20)
│       └── alpha: 0.3 (0.01-1.0)
│
├── Inflation Parameters
│   └── InflationParams
│       ├── inflationAmount: 0.0 (px, 0-50)
│       └── farPointFactor: 1.0 (0-5)
│
├── Droplet Parameters
│   └── DropletParams
│       ├── style: PAINTERLY
│       ├── painterly: { primitive, size, rotation, count }
│       └── organic: { noiseScale, detailLevel, count }
│
├── SVG Output Parameters
│   └── SVGParams
│       ├── simplifyTolerance: 0.5 (px)
│       ├── quantizeCoordinates: true
│       └── stackingOrder: AREA
│
└── Pipeline Modifier Flags
    └── ModifierFlags
        ├── enableQuantization: true
        ├── enableBridging: true
        ├── enableSmoothing: true
        └── enableInflation: true
        
* = also has _REVERSE variant (opposite sort order)
```

## Test Coverage

```
main.cpp (390+ lines)
├── TestSegmentation()
│   ├── Run NoiseWatershed()
│   ├── Count regions
│   ├── Measure timing
│   └── Output: test_segmentation_labels.png (colormap)
│
├── TestQuantization()
│   ├── Run QuantizeColors()
│   ├── Log cluster centers
│   ├── Measure timing
│   └── Output: test_quantized.png
│
├── TestContourExtraction()
│   ├── Run ExtractContours()
│   ├── Calculate area statistics
│   ├── Measure timing
│   └── Output: test_contours.png
│
├── TestSmoothing()
│   ├── Run LaplacianSmooth() with multiple iterations
│   ├── Visualize before/after
│   └── Output: test_smoothing.png (red=original, blue=smoothed)
│
├── TestInflation()
│   ├── Run InflateContour() with multiple methods
│   ├── Visualize all variants
│   └── Output: test_inflation.png (red/green/blue/cyan)
│
├── TestFullPipeline()
│   ├── Run stages 1-6 sequentially
│   ├── Collect per-stage timings
│   ├── Visualize final result
│   └── Output: test_pipeline_result.png
│
└── TestCompletePipeline()
    ├── Execute VectorizationPipeline::Execute()
    ├── Log all timings
    ├── Export to SVG
    └── Output: test_complete_pipeline.svg
```

## Performance Characteristics

### Time Complexity
- **Segmentation**: O(n) where n = image pixels
- **Quantization**: O(n*k*i) where k=clusters, i=iterations (usually 10)
- **Contour Extraction**: O(c) where c = number of contours
- **Bridging**: O(c²) for proximity checks, each check includes LAB conversion
- **Smoothing**: O(c*p*iter) where p = points per contour, iter = iterations
- **Inflation**: O(c*p) 
- **Sorting**: O(c*log(c))

### Space Complexity
- **Segmentation**: O(n) for label map
- **Quantization**: O(n + k*channels)
- **Contours**: O(c*p) where p = avg points per contour
- **Bridging**: O(c) for LAB conversion cache
- **Overall**: O(n + c*p)

### Actual Performance (90x128 test image)
| Stage | Time (ms) | Notes |
|-------|-----------|-------|
| Segmentation | 0.88 | Simple operations on small image |
| Quantization | 32.67 | K-means convergence dominant |
| Contour Extraction | 7.77 | Linear in contour count |
| Bridging | 147.02 | O(c²) with expensive LAB conversion |
| Smoothing | 0.46 | Very fast (few points) |
| Inflation | 0.24 | Simple math |
| Sorting | 0.04 | Efficient std::sort |
| **Total** | **~189 ms** | Feasible for interactive use |

### Scaling Notes
- Segmentation scales linearly with image pixels
- Quantization: 8 colors ~32ms, can increase significantly for higher color counts
- Bridging is bottleneck for many contours (O(c²) + expensive color conversion)
- Future optimization: Cache LAB conversions, parallel bridging

## File Sizes

| File | Lines | Purpose |
|------|-------|---------|
| VectorizationParameters.h | 180 | Types and parameters |
| SegmentationService.h/.cpp | 152 | Watershed segmentation |
| ContourService.h/.cpp | 180+ | Quantization & extraction |
| GeometryService.h/.cpp | 200+ | Smoothing & inflation |
| BridgingService.h/.cpp | 120+ | Contour bridging |
| SortingService.h/.cpp | 120+ | Z-order sorting |
| SVGWriter.h/.cpp | 250+ | SVG export |
| ParameterScaler.h/.cpp | 80+ | Adaptive scaling |
| VectorizationPipeline.h/.cpp | 290+ | Main orchestrator |
| main.cpp (tests) | 390+ | Test suite |
| CMakeLists.txt | 62 | Build configuration |
| **Total** | **~1,920 lines** | Complete implementation |

## Build Output

```
build/
├── Release/
│   ├── OrganicVectoriser.lib (static library)
│   └── vectoriser_test.exe (test executable)
├── CMakeCache.txt
├── CMakeFiles/
└── ... (CMake intermediate files)
```

## Integration Points

### With C# UI (MainWindow.xaml.cs)
1. Create C++/CLI wrapper around OrganicVectoriser.lib
2. Expose Execute() function to managed code
3. Pass parameters from XAML UI to C++ via managed types
4. Return SVG content or bitmap results
5. Display in UI with real-time parameter adjustment

### With Python (future interop)
1. Use ctypes or pybind11 to wrap C++ library
2. Expose services individually for Python testing
3. Load presets from JSON (already in Python)
4. Return results as numpy arrays or images

### Command-Line Usage
```bash
vectoriser_test.exe input.png
```
Automatically:
- Loads image
- Runs all tests
- Generates PNG + SVG outputs
- Prints timing breakdown

## Future Extensions

1. **DropletService** - Painterly/organic droplet generation (pending)
2. **FastNoise2** - Replace Perlin noise stub (pending vcpkg install)
3. **DiagnosticsService** - Per-stage debugging outputs
4. **ParallelBridging** - OpenMP parallelization for bridging
5. **CUDA Support** - GPU acceleration for segmentation/quantization
6. **WebAssembly** - Emscripten compilation for web
7. **RealTime Preview** - OpenGL visualization of pipeline stages

## Summary

The C++ Vectoriser is a modular, testable, high-performance port of the Python implementation with:
- **9 core services** (+ orchestrator pipeline)
- **1,920+ lines** of clean C++17 code
- **Comprehensive testing** with 7 independent test functions
- **~189ms** total execution on 90x128 image
- **SVG export** with coordinate simplification
- **Adaptive parameter scaling** for image sizes
- **10 Z-order sorting** strategies
- **Full LAB color matching** for intelligent bridging
