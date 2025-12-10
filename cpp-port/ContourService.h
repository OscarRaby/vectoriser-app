#pragma once

#include "VectorizationParameters.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace Vectoriser {

class ContourService {
public:
    // Extract contours from watershed label map
    // Returns vector of ContourData with points, centroids, and colors
    static std::vector<ContourData> ExtractContours(
        const cv::Mat& labels,
        const cv::Mat& quantizedImage
    );
    
    // Color quantization using K-means clustering
    // Returns quantized image (CV_8UC3) and cluster centers
    static std::pair<cv::Mat, cv::Mat> QuantizeColors(
        const cv::Mat& image,
        int maxColors
    );
    
    // Calculate contour area using shoelace formula
    static double CalculateArea(const std::vector<cv::Point2d>& contour);
    
    // Calculate luminance from RGB color (ITU-R BT.709)
    static double CalculateLuminance(const cv::Vec3b& color);
    
private:
    // Find most frequent color in region
    static cv::Vec3b FindDominantColor(
        const cv::Mat& quantizedImage,
        const cv::Mat& mask
    );
    
    // Calculate centroid from contour points
    static cv::Point2d CalculateCentroid(const std::vector<cv::Point2d>& contour);
};

} // namespace Vectoriser
