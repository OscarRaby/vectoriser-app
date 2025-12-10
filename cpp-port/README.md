# Organic Vectoriser - C++ Port

Modular C++ implementation of the organic segmentation vectoriser, ported from Python.

## Architecture

This port follows a **modular, testable design** where each image processing stage can be independently:
- Enabled/disabled via `ModifierFlags`
- Tested in isolation via `DiagnosticsService`
- Configured via nested parameter structs

### Core Modules

1. **SegmentationService** - Noise-based watershed segmentation
   - Morphological local minima detection
   - Perlin noise generation (optional FastNoise2)
   - Elevation map creation and watershed

2. **ContourService** - Contour extraction and quantization
   - K-means color quantization
   - Contour finding with dominant color detection
   - Area and luminance calculations

3. **GeometryService** - Geometric transformations
   - Laplacian smoothing
   - Exponential radial inflation
   - Distance calculations

4. **BridgingService** - Contour bridging (TODO)
   - KDTree spatial queries (nanoflann)
   - LAB color space comparison
   - Cosine falloff displacement

5. **DropletService** - Droplet generation (TODO)
   - Painterly droplets (ellipse/rect/polygon)
   - Organic droplets (Perlin-modulated)
   - Brightness filtering

6. **SortingService** - Z-order sorting (TODO)
   - 10 sorting strategies
   - Area/brightness/position-based

7. **SVGWriter** - SVG file generation (TODO)
   - Path simplification (Douglas-Peucker)
   - Native primitive output
   - Coordinate quantization

8. **VectorizationPipeline** - Main orchestrator (TODO)
   - Stage chaining with modifier checks
   - Timing and diagnostics
   - Result aggregation

## Building

### Requirements
- CMake 3.15+
- OpenCV 4.x
- C++17 compiler
- (Optional) FastNoise2 for Perlin noise

### Build Steps

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### With FastNoise2 Support

```bash
cmake -DUSE_FASTNOISE=ON ..
cmake --build .
```

## Usage Example

```cpp
#include "VectorizationPipeline.h"

// Configure parameters
Vectoriser::VectorizationParameters params;
params.segmentation.noiseScale = 60.0;
params.segmentation.blurSigma = 2.0;
params.modifiers.enableBridging = true;
params.modifiers.enableSmoothing = true;

// Load image
cv::Mat image = cv::imread("input.png");

// Execute pipeline
Vectoriser::VectorizationPipeline pipeline;
auto result = pipeline.Execute(image, params);

if (result.success) {
    std::cout << "Generated " << result.contours.size() << " contours\n";
    std::cout << "Execution time: " << result.totalExecutionTime << " ms\n";
}
```

## Parameter Matching with Python

This C++ port maintains **exact parameter compatibility** with the Python version:

| Python Parameter | C++ Parameter | Range | Default |
|-----------------|---------------|-------|---------|
| Noise Scale | `segmentation.noiseScale` | 0-1000 | 60.0 |
| Blur Sigma | `segmentation.blurSigma` | 0.1-10 | 2.0 |
| Compactness | `segmentation.compactness` | 0.0001-1 | 0.001 |
| Max Colors | `quantization.maxColors` | 2-32 | 8 |
| Bridge Distance | `bridging.bridgeDistance` | 0-100 | 5.0 |
| Smooth Iterations | `smoothing.iterations` | 1-20 | 3 |
| Smooth Alpha | `smoothing.alpha` | 0.01-1 | 0.3 |
| Blob Inflation | `inflation.inflationAmount` | 0-50 | 0.0 |

## Algorithm Differences from Python

### Watershed Compactness
- **Python (skimage)**: Supports `compactness` parameter for spatial regularization
- **C++ (OpenCV)**: `cv::watershed()` doesn't support compactness
- **Impact**: Minor boundary smoothness differences

### Perlin Noise
- **Python**: Uses `noise` package (optional, fallback to zeros)
- **C++**: Uses FastNoise2 (optional, fallback to zeros)
- **Impact**: None if both unavailable, identical when available

## Testing

Each module can be tested independently via `DiagnosticsService`:

```cpp
#include "DiagnosticsService.h"

Vectoriser::DiagnosticsService diag;
diag.TestSegmentation(image, params);  // Outputs: noise, elevation, minima, labels
diag.TestBridging(contours, params);   // Outputs: bridged contours visualization
```

## Implementation Status

- [x] Parameter structures
- [x] Segmentation service
- [x] Contour service
- [x] Geometry service (smoothing, inflation)
- [ ] Bridging service (KDTree integration needed)
- [ ] Droplet service
- [ ] Sorting service
- [ ] SVG writer
- [ ] Main pipeline orchestrator
- [ ] Diagnostics service
- [ ] Example application

## License

Same as parent project.
