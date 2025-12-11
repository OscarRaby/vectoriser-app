#pragma once

#include "VectorizationParameters.h"
#include <opencv2/opencv.hpp>

namespace Vectoriser {

class ParameterScaler {
public:
    // Calculate scale factor based on image size
    // Matches Python: get_scale_factor()
    static double GetScaleFactor(
        cv::Size imageSize,
        const SegmentationParams& params
    );
    
    // Scale segmentation parameters for image size
    static SegmentationParams ScaleParameters(
        const SegmentationParams& params,
        cv::Size imageSize
    );
    
private:
    // Ensure minimum value to avoid division by zero
    static double EnsureMinimum(double value, double minimum = 1e-6);
};

} // namespace Vectoriser
