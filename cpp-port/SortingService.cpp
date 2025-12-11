#include "SortingService.h"
#include "ContourService.h"
#include <algorithm>
#include <numeric>

namespace Vectoriser {

double SortingService::CalculateLuminance(const cv::Vec3b& color) {
    return ContourService::CalculateLuminance(color);
}

double SortingService::CalculateDistanceFromCenter(
    const cv::Point2d& centroid,
    const cv::Point2d& center
) {
    return cv::norm(centroid - center);
}

std::vector<size_t> SortingService::SortContoursByOrder(
    const std::vector<ContourData>& contours,
    StackingOrder order,
    cv::Size imageSize
) {
    // Initialize indices
    std::vector<size_t> indices(contours.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Sort based on order type
    switch (order) {
        case StackingOrder::AREA:
            // Largest first
            std::sort(indices.begin(), indices.end(),
                [&contours](size_t a, size_t b) {
                    return contours[a].area > contours[b].area;
                });
            break;
            
        case StackingOrder::AREA_REVERSE:
            // Smallest first
            std::sort(indices.begin(), indices.end(),
                [&contours](size_t a, size_t b) {
                    return contours[a].area < contours[b].area;
                });
            break;
            
        case StackingOrder::BRIGHTNESS:
            // Darkest first
            std::sort(indices.begin(), indices.end(),
                [&contours](size_t a, size_t b) {
                    return CalculateLuminance(contours[a].color) < 
                           CalculateLuminance(contours[b].color);
                });
            break;
            
        case StackingOrder::BRIGHTNESS_REVERSE:
            // Brightest first
            std::sort(indices.begin(), indices.end(),
                [&contours](size_t a, size_t b) {
                    return CalculateLuminance(contours[a].color) > 
                           CalculateLuminance(contours[b].color);
                });
            break;
            
        case StackingOrder::POSITION_X:
            // Leftmost first
            std::sort(indices.begin(), indices.end(),
                [&contours](size_t a, size_t b) {
                    return contours[a].centroid.x < contours[b].centroid.x;
                });
            break;
            
        case StackingOrder::POSITION_X_REVERSE:
            // Rightmost first
            std::sort(indices.begin(), indices.end(),
                [&contours](size_t a, size_t b) {
                    return contours[a].centroid.x > contours[b].centroid.x;
                });
            break;
            
        case StackingOrder::POSITION_Y:
            // Topmost first (row coordinate, smaller = top)
            std::sort(indices.begin(), indices.end(),
                [&contours](size_t a, size_t b) {
                    return contours[a].centroid.y < contours[b].centroid.y;
                });
            break;
            
        case StackingOrder::POSITION_Y_REVERSE:
            // Bottommost first
            std::sort(indices.begin(), indices.end(),
                [&contours](size_t a, size_t b) {
                    return contours[a].centroid.y > contours[b].centroid.y;
                });
            break;
            
        case StackingOrder::POSITION_CENTRE:
            // Farthest from center first
            {
                cv::Point2d center(imageSize.height / 2.0, imageSize.width / 2.0);
                std::sort(indices.begin(), indices.end(),
                    [&contours, &center](size_t a, size_t b) {
                        return CalculateDistanceFromCenter(contours[a].centroid, center) >
                               CalculateDistanceFromCenter(contours[b].centroid, center);
                    });
            }
            break;
            
        case StackingOrder::POSITION_CENTRE_REVERSE:
            // Closest to center first
            {
                cv::Point2d center(imageSize.height / 2.0, imageSize.width / 2.0);
                std::sort(indices.begin(), indices.end(),
                    [&contours, &center](size_t a, size_t b) {
                        return CalculateDistanceFromCenter(contours[a].centroid, center) <
                               CalculateDistanceFromCenter(contours[b].centroid, center);
                    });
            }
            break;
    }
    
    return indices;
}

} // namespace Vectoriser
