#pragma once

#include "VectorizationParameters.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace Vectoriser {

class SortingService {
public:
    // Sort contours by specified stacking order
    // Returns sorted indices
    static std::vector<size_t> SortContoursByOrder(
        const std::vector<ContourData>& contours,
        StackingOrder order,
        cv::Size imageSize
    );
    
private:
    // Calculate luminance from RGB (ITU-R BT.709)
    static double CalculateLuminance(const cv::Vec3b& color);
    
    // Calculate distance from image center
    static double CalculateDistanceFromCenter(
        const cv::Point2d& centroid,
        const cv::Point2d& center
    );
};

} // namespace Vectoriser
