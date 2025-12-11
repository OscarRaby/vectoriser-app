#pragma once

#include "VectorizationParameters.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace Vectoriser {

class BridgingService {
public:
    // Improved bridge contours - connects similar-color neighbors
    // Matches Python: improved_bridge_contours()
    static std::vector<std::vector<cv::Point2d>> ImprovedBridgeContours(
        const std::vector<std::vector<cv::Point2d>>& contours,
        const std::vector<cv::Point2d>& centroids,
        const std::vector<cv::Vec3b>& colors,
        const BridgingParams& params
    );
    
private:
    // Convert RGB to LAB color space for perceptual distance
    static cv::Vec3d RGBToLAB(const cv::Vec3b& rgb);
    
    // Calculate LAB color distance
    static double CalculateLABDistance(const cv::Vec3d& lab1, const cv::Vec3d& lab2);
    
    // Calculate angle between two vectors (for curvature check)
    static double CalculateAngle(const cv::Point2d& v1, const cv::Point2d& v2);
    
    // Apply bridging to a single contour
    static std::vector<cv::Point2d> BridgeContour(
        const std::vector<cv::Point2d>& contour,
        const cv::Point2d& centroid,
        const cv::Vec3d& colorLAB,
        const std::vector<cv::Point2d>& allCentroids,
        const std::vector<cv::Vec3d>& allColorsLAB,
        const BridgingParams& params,
        int currentIndex
    );
};

} // namespace Vectoriser
