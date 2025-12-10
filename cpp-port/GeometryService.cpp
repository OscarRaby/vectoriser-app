#include "GeometryService.h"
#include <cmath>

namespace Vectoriser {

std::vector<cv::Point2d> GeometryService::LaplacianSmooth(
    const std::vector<cv::Point2d>& contour,
    int iterations,
    double alpha
) {
    if (contour.size() < 3) return contour;
    
    std::vector<cv::Point2d> result = contour;
    int n = contour.size();
    
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<cv::Point2d> newContour = result;
        
        for (int i = 0; i < n; i++) {
            int prevIdx = (i - 1 + n) % n;
            int nextIdx = (i + 1) % n;
            
            cv::Point2d prevPt = result[prevIdx];
            cv::Point2d nextPt = result[nextIdx];
            
            // new_pt = (1-alpha)*pt + alpha*0.5*(prev+next)
            newContour[i] = (1.0 - alpha) * result[i] + 
                           alpha * 0.5 * (prevPt + nextPt);
        }
        
        result = std::move(newContour);
    }
    
    return result;
}

std::vector<double> GeometryService::CalculateNormalizedDistances(
    const std::vector<cv::Point2d>& contour,
    const cv::Point2d& centroid
) {
    std::vector<double> distances;
    distances.reserve(contour.size());
    
    double maxDistance = 0.0;
    
    // Calculate distances
    for (const auto& pt : contour) {
        cv::Point2d vec = pt - centroid;
        double dist = cv::norm(vec);
        distances.push_back(dist);
        maxDistance = std::max(maxDistance, dist);
    }
    
    // Normalize
    if (maxDistance > 0.0) {
        for (auto& dist : distances) {
            dist /= maxDistance;
        }
    }
    
    return distances;
}

double GeometryService::CalculateStackingScale(size_t orderIdx, size_t totalCount) {
    if (totalCount <= 1) return 1.0;
    return static_cast<double>(totalCount - orderIdx) / static_cast<double>(totalCount);
}

std::vector<cv::Point2d> GeometryService::InflateContour(
    const std::vector<cv::Point2d>& contour,
    double inflationAmount,
    double farPointFactor
) {
    if (contour.empty()) return contour;
    
    // Calculate centroid
    cv::Point2d centroid(0, 0);
    for (const auto& pt : contour) {
        centroid += pt;
    }
    centroid *= (1.0 / contour.size());
    
    // Calculate normalized distances
    std::vector<double> normDistances = CalculateNormalizedDistances(contour, centroid);
    
    // Apply inflation
    std::vector<cv::Point2d> inflated;
    inflated.reserve(contour.size());
    
    for (size_t i = 0; i < contour.size(); i++) {
        cv::Point2d vec = contour[i] - centroid;
        double dist = cv::norm(vec);
        
        // Exponential inflation: amount * exp((farFactor-1) * normalized_distance)
        double inflationFactor = inflationAmount * 
            std::exp((farPointFactor - 1.0) * normDistances[i]);
        
        cv::Point2d direction = (dist > 0.0) ? (vec / dist) : cv::Point2d(0, 0);
        inflated.push_back(contour[i] + direction * inflationFactor);
    }
    
    return inflated;
}

} // namespace Vectoriser
