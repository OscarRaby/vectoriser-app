#include "SegmentationService.h"
#include <iostream>

namespace Vectoriser {

cv::Mat SegmentationService::RGBToGrayFloat(const cv::Mat& rgb) {
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
    gray.convertTo(gray, CV_32FC1, 1.0 / 255.0);
    return gray;
}

cv::Mat SegmentationService::GeneratePerlinNoise(int width, int height, double scale) {
    // TODO: Integrate FastNoise2 or similar library
    // For now, return zero-filled Mat as fallback
    // This matches Python behavior when 'noise' package is unavailable
    cv::Mat noise = cv::Mat::zeros(height, width, CV_32FC1);
    
    #ifdef USE_FASTNOISE
    // FastNoise2 implementation would go here
    // Example:
    // auto fnSimplex = FastNoise::New<FastNoise::Simplex>();
    // auto fnFractal = FastNoise::New<FastNoise::FractalFBm>();
    // fnFractal->SetSource(fnSimplex);
    // fnFractal->SetOctaveCount(3);
    // 
    // std::vector<float> noiseData(width * height);
    // fnFractal->GenUniformGrid2D(noiseData.data(), 0, 0, width, height, 1.0f / scale, seed);
    // 
    // for (int y = 0; y < height; y++) {
    //     for (int x = 0; x < width; x++) {
    //         noise.at<float>(y, x) = noiseData[y * width + x];
    //     }
    // }
    #endif
    
    return noise;
}

cv::Mat SegmentationService::CreateElevationMap(
    const cv::Mat& blurred,
    const cv::Mat& noise
) {
    cv::Mat elevation;
    cv::addWeighted(blurred, 0.7, noise, 0.3, 0.0, elevation);
    return elevation;
}

cv::Mat SegmentationService::MorphologicalLocalMinima(const cv::Mat& elevationMap) {
    // Python: local_min = morphology.local_minima(elevation_map)
    // A pixel is a local minimum if it's <= all neighbors
    // Strategy: dilate finds max in neighborhood, compare to find where unchanged
    
    cv::Mat localMin;
    cv::Mat dilated;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    
    // Dilate finds maximum in 3x3 neighborhood
    cv::dilate(elevationMap, dilated, kernel);
    
    // Local minima are where elevation == dilated (minimum in neighborhood)
    cv::compare(elevationMap, dilated, localMin, cv::CMP_EQ);
    
    return localMin;
}

cv::Mat SegmentationService::NoiseWatershed(
    const cv::Mat& rgb,
    double noiseScale,
    double blurSigma,
    double compactness
) {
    // Validate input
    if (rgb.type() != CV_8UC3) {
        throw std::invalid_argument("Expected CV_8UC3 input for NoiseWatershed");
    }
    
    // 1. Convert to grayscale float [0, 1]
    cv::Mat gray = RGBToGrayFloat(rgb);
    
    // 2. Gaussian blur
    cv::Mat blurred;
    int ksize = static_cast<int>(blurSigma * 6) | 1;  // Ensure odd
    cv::GaussianBlur(gray, blurred, cv::Size(ksize, ksize), blurSigma);
    
    // 3. Generate Perlin noise field
    cv::Mat noise = GeneratePerlinNoise(rgb.cols, rgb.rows, noiseScale);
    
    // 4. Create elevation map: 0.7 * blurred + 0.3 * noise
    cv::Mat elevation = CreateElevationMap(blurred, noise);
    
    // 5. Find local minima using morphological approach
    cv::Mat localMin = MorphologicalLocalMinima(elevation);
    
    // 6. Label connected components (markers for watershed)
    cv::Mat markers;
    cv::connectedComponents(localMin, markers, 8, CV_32S);
    
    // 7. Apply watershed
    // Note: OpenCV's watershed doesn't have compactness parameter
    // The compactness parameter from Python's skimage is ignored here
    cv::Mat rgbCopy = rgb.clone();
    cv::watershed(rgbCopy, markers);
    
    return markers;
}

} // namespace Vectoriser
