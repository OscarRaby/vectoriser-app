#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace Vectoriser {

class GeometryService {
public:
    // Laplacian smoothing: new_pt = (1-alpha)*pt + alpha*0.5*(prev+next)
    static std::vector<cv::Point2d> LaplacianSmooth(
        const std::vector<cv::Point2d>& contour,
        int iterations,
        double alpha
    );
    
    // Radial inflation with exponential far-point scaling
    // inflated = point + direction * amount * exp((farFactor-1) * normalized_distance)
    static std::vector<cv::Point2d> InflateContour(
        const std::vector<cv::Point2d>& contour,
        double inflationAmount,
        double farPointFactor
    );
    
    // Calculate normalized distance from centroid
    static std::vector<double> CalculateNormalizedDistances(
        const std::vector<cv::Point2d>& contour,
        const cv::Point2d& centroid
    );
    
    // Calculate stacking scale factor
    // stack_scale = (N_total - order_idx) / N_total
    static double CalculateStackingScale(size_t orderIdx, size_t totalCount);
};

} // namespace Vectoriser
