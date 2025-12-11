#include "ParameterScaler.h"
#include <algorithm>
#include <cmath>

namespace Vectoriser {

double ParameterScaler::EnsureMinimum(double value, double minimum) {
    return std::max(value, minimum);
}

double ParameterScaler::GetScaleFactor(
    cv::Size imageSize,
    const SegmentationParams& params
) {
    int h = imageSize.height;
    int w = imageSize.width;
    int refH = params.referenceCropSize.height;
    int refW = params.referenceCropSize.width;
    
    switch (params.scalingMethod) {
        case ScalingMethod::MAX:
            return static_cast<double>(std::max(h, w)) / std::max(refH, refW);
            
        case ScalingMethod::MIN:
            return static_cast<double>(std::min(h, w)) / std::min(refH, refW);
            
        case ScalingMethod::AVERAGE:
            return ((h + w) / 2.0) / ((refH + refW) / 2.0);
            
        case ScalingMethod::AREA:
            return static_cast<double>(h * w) / (refH * refW);
            
        case ScalingMethod::SQRT_AREA:
            return std::sqrt(static_cast<double>(h * w) / (refH * refW));
            
        default:
            return 1.0;
    }
}

SegmentationParams ParameterScaler::ScaleParameters(
    const SegmentationParams& params,
    cv::Size imageSize
) {
    double scaleFactor = GetScaleFactor(imageSize, params);
    double multiplier = EnsureMinimum(params.segmentationMultiplier, 1e-6);
    
    SegmentationParams scaled = params;
    
    // Scale parameters: param * scaleFactor / multiplier
    scaled.noiseScale = EnsureMinimum(params.noiseScale * scaleFactor / multiplier);
    scaled.blurSigma = EnsureMinimum(params.blurSigma * scaleFactor / multiplier);
    scaled.compactness = EnsureMinimum(params.compactness * scaleFactor / multiplier);
    
    return scaled;
}

} // namespace Vectoriser
