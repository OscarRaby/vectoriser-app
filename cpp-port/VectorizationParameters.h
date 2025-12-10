#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace Vectoriser {

// Scaling methods for parameter adaptation to image size
enum class ScalingMethod {
    MAX,           // max(w,h) / max(ref_w, ref_h)
    MIN,           // min(w,h) / min(ref_w, ref_h)
    AVERAGE,       // average of dimensions
    AREA,          // (w*h) / (ref_w * ref_h)
    SQRT_AREA      // sqrt((w*h) / (ref_w * ref_h))
};

// Z-order stacking strategies
enum class StackingOrder {
    AREA,
    AREA_REVERSE,
    BRIGHTNESS,
    BRIGHTNESS_REVERSE,
    POSITION_X,
    POSITION_X_REVERSE,
    POSITION_Y,
    POSITION_Y_REVERSE,
    POSITION_CENTRE,
    POSITION_CENTRE_REVERSE
};

// Droplet generation styles
enum class DropletStyle {
    PAINTERLY,     // Regular ellipses/rects
    ORGANIC        // Perlin-noise modulated
};

// Painterly primitive types for SVG output
enum class PainterlyPrimitive {
    ELLIPSE,       // Native SVG <ellipse>
    RECT,          // Native SVG <rect>
    POLYGON        // Polygon approximation as <path>
};

// Segmentation parameters
struct SegmentationParams {
    double noiseScale = 60.0;        // Perlin noise scale (0-1000)
    double blurSigma = 2.0;          // Gaussian blur sigma (0.1-10)
    double compactness = 0.001;      // Watershed compactness (0.0001-1)
    
    // Scaling configuration
    ScalingMethod scalingMethod = ScalingMethod::MAX;
    double segmentationMultiplier = 1.0;  // Divider for parameters (0.1-10)
    cv::Size referenceCropSize = cv::Size(150, 150);  // Reference for scaling
};

// Color quantization parameters
struct QuantizationParams {
    int maxColors = 8;               // K-means clusters (2-32)
};

// Contour bridging parameters
struct BridgingParams {
    double bridgeDistance = 5.0;     // Push distance toward neighbors (0-100)
    double colorTolerance = 10.0;    // LAB color difference threshold (0-100)
    double proximityThreshold = 50.0; // Search radius (0-200)
    int falloffRadius = 5;           // Neighbor points affected (1-30)
    double maxCurvature = 160.0;     // Max angle in degrees (1-360)
};

// Laplacian smoothing parameters
struct SmoothingParams {
    int iterations = 3;              // Smoothing passes (1-20)
    double alpha = 0.3;              // Smoothing strength (0.01-1.0)
};

// Contour inflation parameters
struct InflationParams {
    double inflationAmount = 0.0;    // Base expansion (0-50)
    double farPointFactor = 1.0;     // Exponential far-point multiplier (0-5)
    bool proportionalToStacking = true; // Scale by z-order
};

// Painterly droplet parameters
struct PainterlyDropletParams {
    int density = 0;                 // Droplets per blob (0-10)
    double minDistance = 5.0;        // Min distance from blob (0-100)
    double maxDistance = 15.0;       // Max distance from blob (0-200)
    double sizeMean = 3.0;           // Mean droplet size (1-50)
    double sizeStd = 1.0;            // Size standard deviation (0-20)
    double spreadAngle = 5.0;        // Angular spread in degrees (0-90)
    double globalRotation = 0.0;     // Global rotation offset in degrees (-180-180)
    bool rectHorizontal = false;     // Force rectangles to be horizontal
    bool useSVGPrimitives = false;   // Use native <ellipse>/<rect> instead of <path>
    PainterlyPrimitive primitiveType = PainterlyPrimitive::ELLIPSE;
};

// Organic droplet parameters
struct OrganicDropletParams {
    double minBrightness = 128.0;    // Brightness threshold (0-255)
    int density = 3;                 // Droplets per neighbor (0-20)
    double strength = 1.0;           // Perlin noise amplitude (0-5)
    double jitter = 0.5;             // Random noise (0-5)
    double elongation = 0.0;         // Axis scaling (-2 to 5)
    double percentPerBlob = 100.0;   // Neighbor sampling percentage (0-100)
};

// Combined droplet parameters
struct DropletParams {
    DropletStyle style = DropletStyle::PAINTERLY;
    PainterlyDropletParams painterly;
    OrganicDropletParams organic;
};

// SVG output parameters
struct SVGParams {
    std::string outputPath = "output.svg";
    double simplifyTolerance = 0.5;  // Douglas-Peucker tolerance in pixels (0-5)
    bool quantizeCoordinates = true; // Round to integers for smaller files
};

// Pipeline modifier flags
struct ModifierFlags {
    bool enableQuantization = true;
    bool enableBridging = true;
    bool enableSmoothing = true;
    bool enableInflation = true;
};

// Complete vectorization parameters
struct VectorizationParameters {
    SegmentationParams segmentation;
    QuantizationParams quantization;
    BridgingParams bridging;
    SmoothingParams smoothing;
    InflationParams inflation;
    DropletParams droplets;
    SVGParams svg;
    ModifierFlags modifiers;
    StackingOrder stackingOrder = StackingOrder::AREA;
};

// Droplet descriptor for output
struct DropletDescriptor {
    enum class Type { ELLIPSE, RECT, POLYGON } type;
    cv::Point2d center;              // Center point (x, y in SVG coords)
    cv::Point2d size;                // For ellipse: (rx, ry), for rect: (w, h)
    double rotation = 0.0;           // Rotation angle in degrees
    std::vector<cv::Point2d> polygon; // For POLYGON type
    cv::Vec3b color;                 // RGB color
};

// Contour with metadata
struct ContourData {
    std::vector<cv::Point2d> points; // Contour points (row, col) format
    cv::Point2d centroid;            // Centroid
    cv::Vec3b color;                 // Dominant color
    double area;                     // Contour area
};

// Pipeline execution result
struct PipelineResult {
    std::vector<ContourData> contours;
    std::vector<DropletDescriptor> droplets;
    std::vector<size_t> zOrderIndices;  // Sorted indices
    double totalExecutionTime = 0.0;     // Milliseconds
    
    // Stage-by-stage timing
    double segmentationTime = 0.0;
    double quantizationTime = 0.0;
    double bridgingTime = 0.0;
    double smoothingTime = 0.0;
    double inflationTime = 0.0;
    double dropletTime = 0.0;
    
    bool success = false;
    std::string errorMessage;
};

} // namespace Vectoriser
