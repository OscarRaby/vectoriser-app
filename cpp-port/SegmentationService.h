#pragma once

#include "VectorizationParameters.h"
#include <opencv2/opencv.hpp>
#include <memory>

namespace Vectoriser {

class SegmentationService {
public:
    // Main noise-based watershed segmentation
    // Returns label map where each region has unique integer ID
    static cv::Mat NoiseWatershed(
        const cv::Mat& rgb,
        double noiseScale,
        double blurSigma,
        double compactness
    );
    
    // Generate Perlin noise field
    // Returns float32 Mat normalized to [0, 1]
    static cv::Mat GeneratePerlinNoise(
        int width,
        int height,
        double scale
    );
    
    // Find local minima using morphological approach
    // Matches Python: morphology.local_minima(elevation_map)
    // Returns binary mask where local minima = 255
    static cv::Mat MorphologicalLocalMinima(const cv::Mat& elevationMap);
    
    // Create elevation map from blurred grayscale and noise
    // Returns: 0.7 * blurred + 0.3 * noise
    static cv::Mat CreateElevationMap(
        const cv::Mat& blurred,
        const cv::Mat& noise
    );

private:
    // Helper: convert RGB to grayscale float [0, 1]
    static cv::Mat RGBToGrayFloat(const cv::Mat& rgb);
};

} // namespace Vectoriser
