#include "BridgingService.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Vectoriser {

cv::Vec3d BridgingService::RGBToLAB(const cv::Vec3b& rgb) {
    // Convert RGB to LAB using OpenCV
    cv::Mat rgbMat(1, 1, CV_8UC3, cv::Scalar(rgb[0], rgb[1], rgb[2]));
    cv::Mat labMat;
    cv::cvtColor(rgbMat, labMat, cv::COLOR_RGB2Lab);
    
    cv::Vec3b labVec = labMat.at<cv::Vec3b>(0, 0);
    return cv::Vec3d(labVec[0], labVec[1], labVec[2]);
}

double BridgingService::CalculateLABDistance(const cv::Vec3d& lab1, const cv::Vec3d& lab2) {
    cv::Vec3d diff = lab1 - lab2;
    return cv::norm(diff);
}

double BridgingService::CalculateAngle(const cv::Point2d& v1, const cv::Point2d& v2) {
    double norm1 = cv::norm(v1);
    double norm2 = cv::norm(v2);
    
    if (norm1 == 0.0 || norm2 == 0.0) return 0.0;
    
    double cosAngle = v1.dot(v2) / (norm1 * norm2);
    cosAngle = std::max(-1.0, std::min(1.0, cosAngle)); // Clamp to [-1, 1]
    
    return std::acos(cosAngle);
}

std::vector<cv::Point2d> BridgingService::BridgeContour(
    const std::vector<cv::Point2d>& contour,
    const cv::Point2d& centroid,
    const cv::Vec3d& colorLAB,
    const std::vector<cv::Point2d>& allCentroids,
    const std::vector<cv::Vec3d>& allColorsLAB,
    const BridgingParams& params,
    int currentIndex
) {
    std::vector<cv::Point2d> result = contour;
    int n = result.size();
    
    // Find neighbors within proximity threshold
    for (size_t j = 0; j < allCentroids.size(); j++) {
        if (static_cast<int>(j) == currentIndex) continue;
        
        double distance = cv::norm(centroid - allCentroids[j]);
        if (distance > params.proximityThreshold) continue;
        
        // Check color similarity in LAB space
        double colorDiff = CalculateLABDistance(colorLAB, allColorsLAB[j]);
        if (colorDiff > params.colorTolerance) continue;
        
        // Find closest point on contour to this neighbor
        double minDist = std::numeric_limits<double>::max();
        int closestIdx = 0;
        
        for (int i = 0; i < n; i++) {
            double dist = cv::norm(result[i] - allCentroids[j]);
            if (dist < minDist) {
                minDist = dist;
                closestIdx = i;
            }
        }
        
        // Calculate direction toward neighbor
        cv::Point2d direction = allCentroids[j] - result[closestIdx];
        double norm = cv::norm(direction);
        if (norm == 0.0) continue;
        direction /= norm;
        
        // Apply displacement with cosine falloff
        for (int offset = -params.falloffRadius; offset <= params.falloffRadius; offset++) {
            int neighborIdx = (closestIdx + offset + n) % n;
            int prevIdx = (neighborIdx - 1 + n) % n;
            int nextIdx = (neighborIdx + 1) % n;
            
            // Check curvature at this point
            cv::Point2d v1 = result[neighborIdx] - result[prevIdx];
            cv::Point2d v2 = result[nextIdx] - result[neighborIdx];
            
            double angle = CalculateAngle(v1, v2);
            double maxCurvatureRad = params.maxCurvature * M_PI / 180.0;
            
            if (angle > maxCurvatureRad) continue;
            
            // Cosine falloff weight
            double weight = 0.5 * (1.0 + std::cos(M_PI * offset / params.falloffRadius));
            cv::Point2d displacement = direction * params.bridgeDistance * weight;
            result[neighborIdx] += displacement;
        }
    }
    
    return result;
}

std::vector<std::vector<cv::Point2d>> BridgingService::ImprovedBridgeContours(
    const std::vector<std::vector<cv::Point2d>>& contours,
    const std::vector<cv::Point2d>& centroids,
    const std::vector<cv::Vec3b>& colors,
    const BridgingParams& params
) {
    // Convert all colors to LAB space
    std::vector<cv::Vec3d> colorsLAB;
    colorsLAB.reserve(colors.size());
    for (const auto& color : colors) {
        colorsLAB.push_back(RGBToLAB(color));
    }
    
    // Bridge each contour
    std::vector<std::vector<cv::Point2d>> bridged;
    bridged.reserve(contours.size());
    
    for (size_t i = 0; i < contours.size(); i++) {
        auto bridgedContour = BridgeContour(
            contours[i],
            centroids[i],
            colorsLAB[i],
            centroids,
            colorsLAB,
            params,
            static_cast<int>(i)
        );
        bridged.push_back(std::move(bridgedContour));
    }
    
    return bridged;
}

} // namespace Vectoriser
