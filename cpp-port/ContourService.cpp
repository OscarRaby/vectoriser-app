#include "ContourService.h"
#include <map>
#include <algorithm>

namespace Vectoriser {

double ContourService::CalculateLuminance(const cv::Vec3b& color) {
    // ITU-R BT.709 luminance formula
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
}

double ContourService::CalculateArea(const std::vector<cv::Point2d>& contour) {
    // Shoelace formula
    if (contour.size() < 3) return 0.0;
    
    double area = 0.0;
    int n = contour.size();
    
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        // contour is in (row, col) = (y, x) format
        area += contour[i].y * contour[j].x;
        area -= contour[j].y * contour[i].x;
    }
    
    return std::abs(area) * 0.5;
}

cv::Point2d ContourService::CalculateCentroid(const std::vector<cv::Point2d>& contour) {
    if (contour.empty()) return cv::Point2d(0, 0);
    
    cv::Point2d sum(0, 0);
    for (const auto& pt : contour) {
        sum += pt;
    }
    
    return sum * (1.0 / contour.size());
}

cv::Vec3b ContourService::FindDominantColor(
    const cv::Mat& quantizedImage,
    const cv::Mat& mask
) {
    std::map<cv::Vec3b, int, std::less<>> colorCounts;
    
    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            if (mask.at<uchar>(y, x) > 0) {
                cv::Vec3b color = quantizedImage.at<cv::Vec3b>(y, x);
                colorCounts[color]++;
            }
        }
    }
    
    if (colorCounts.empty()) {
        return cv::Vec3b(0, 0, 0);
    }
    
    // Find color with max count
    auto maxIt = std::max_element(
        colorCounts.begin(),
        colorCounts.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
    );
    
    return maxIt->first;
}

std::pair<cv::Mat, cv::Mat> ContourService::QuantizeColors(
    const cv::Mat& image,
    int maxColors
) {
    // Reshape image to pixel array
    cv::Mat pixels = image.reshape(1, image.rows * image.cols);
    pixels.convertTo(pixels, CV_32F);
    
    // K-means clustering
    cv::Mat labels, centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.2);
    cv::kmeans(pixels, maxColors, labels, criteria, 10, cv::KMEANS_PP_CENTERS, centers);
    
    // Create quantized image
    centers.convertTo(centers, CV_8U);
    cv::Mat quantized(image.size(), image.type());
    
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int clusterIdx = labels.at<int>(y * image.cols + x);
            quantized.at<cv::Vec3b>(y, x) = centers.at<cv::Vec3b>(clusterIdx, 0);
        }
    }
    
    return {quantized, centers};
}

std::vector<ContourData> ContourService::ExtractContours(
    const cv::Mat& labels,
    const cv::Mat& quantizedImage
) {
    std::vector<ContourData> result;
    
    // Find unique labels
    double minVal, maxVal;
    cv::minMaxLoc(labels, &minVal, &maxVal);
    
    for (int segId = 0; segId <= static_cast<int>(maxVal); segId++) {
        // Create mask for this segment
        cv::Mat mask = (labels == segId);
        
        // Skip if empty
        if (cv::countNonZero(mask) == 0) continue;
        
        // Find dominant color
        cv::Vec3b mainColor = FindDominantColor(quantizedImage, mask);
        
        // Find contours using OpenCV
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        
        if (contours.empty()) continue;
        
        // Take the largest contour
        auto maxContour = std::max_element(
            contours.begin(),
            contours.end(),
            [](const auto& a, const auto& b) { return a.size() < b.size(); }
        );
        
        // Convert to Point2d (row, col) format
        std::vector<cv::Point2d> contourPoints;
        contourPoints.reserve(maxContour->size());
        for (const auto& pt : *maxContour) {
            // OpenCV contours are in (x, y) format, convert to (row, col) = (y, x)
            contourPoints.emplace_back(pt.y, pt.x);
        }
        
        ContourData data;
        data.points = std::move(contourPoints);
        data.centroid = CalculateCentroid(data.points);
        data.color = mainColor;
        data.area = CalculateArea(data.points);
        
        result.push_back(std::move(data));
    }
    
    return result;
}

} // namespace Vectoriser
