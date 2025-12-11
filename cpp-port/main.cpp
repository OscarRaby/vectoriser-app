#include "VectorizationParameters.h"
#include "SegmentationService.h"
#include "ContourService.h"
#include "GeometryService.h"
#include "BridgingService.h"
#include "SortingService.h"
#include "SVGWriter.h"
#include "ParameterScaler.h"
#include "VectorizationPipeline.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace Vectoriser;

void TestSegmentation(const cv::Mat& image) {
    std::cout << "\n=== Testing Segmentation Service ===\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run noise watershed
    cv::Mat labels = SegmentationService::NoiseWatershed(
        image,
        60.0,  // noiseScale
        2.0,   // blurSigma
        0.001  // compactness
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Count unique labels
    double minVal, maxVal;
    cv::minMaxLoc(labels, &minVal, &maxVal);
    
    std::cout << "  Image size: " << image.cols << "x" << image.rows << "\n";
    std::cout << "  Unique regions: " << static_cast<int>(maxVal) + 1 << "\n";
    std::cout << "  Time: " << duration.count() << " ms\n";
    
    // Visualize labels
    cv::Mat labelViz;
    labels.convertTo(labelViz, CV_8U);
    cv::normalize(labelViz, labelViz, 0, 255, cv::NORM_MINMAX);
    cv::applyColorMap(labelViz, labelViz, cv::COLORMAP_JET);
    cv::imwrite("test_segmentation_labels.png", labelViz);
    std::cout << "  Saved: test_segmentation_labels.png\n";
}

void TestQuantization(const cv::Mat& image) {
    std::cout << "\n=== Testing Color Quantization ===\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto [quantized, centers] = ContourService::QuantizeColors(image, 8);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "  Max colors: 8\n";
    std::cout << "  Time: " << duration.count() << " ms\n";
    std::cout << "  Cluster centers:\n";
    
    for (int i = 0; i < centers.rows; i++) {
        cv::Vec3b color = centers.at<cv::Vec3b>(i, 0);
        std::cout << "    Color " << i << ": RGB(" 
                  << static_cast<int>(color[0]) << ", "
                  << static_cast<int>(color[1]) << ", "
                  << static_cast<int>(color[2]) << ")\n";
    }
    
    cv::imwrite("test_quantized.png", quantized);
    std::cout << "  Saved: test_quantized.png\n";
}

void TestContourExtraction(const cv::Mat& image) {
    std::cout << "\n=== Testing Contour Extraction ===\n";
    
    // First segment
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat labels = SegmentationService::NoiseWatershed(image, 60.0, 2.0, 0.001);
    
    // Then quantize
    auto [quantized, centers] = ContourService::QuantizeColors(image, 8);
    
    // Extract contours
    auto contours = ContourService::ExtractContours(labels, quantized);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "  Extracted contours: " << contours.size() << "\n";
    std::cout << "  Time: " << duration.count() << " ms\n";
    
    // Calculate statistics
    double totalArea = 0.0;
    double minArea = std::numeric_limits<double>::max();
    double maxArea = 0.0;
    
    for (const auto& contour : contours) {
        totalArea += contour.area;
        minArea = std::min(minArea, contour.area);
        maxArea = std::max(maxArea, contour.area);
    }
    
    std::cout << "  Total area: " << totalArea << "\n";
    std::cout << "  Min area: " << minArea << "\n";
    std::cout << "  Max area: " << maxArea << "\n";
    std::cout << "  Avg area: " << (totalArea / contours.size()) << "\n";
    
    // Visualize contours
    cv::Mat visualization = image.clone();
    for (const auto& contourData : contours) {
        std::vector<cv::Point> cvPoints;
        for (const auto& pt : contourData.points) {
            // Convert from (row, col) to (x, y)
            cvPoints.emplace_back(static_cast<int>(pt.y), static_cast<int>(pt.x));
        }
        
        cv::Scalar color(contourData.color[0], contourData.color[1], contourData.color[2]);
        cv::fillPoly(visualization, std::vector<std::vector<cv::Point>>{cvPoints}, color);
    }
    
    cv::imwrite("test_contours.png", visualization);
    std::cout << "  Saved: test_contours.png\n";
}

void TestSmoothing() {
    std::cout << "\n=== Testing Laplacian Smoothing ===\n";
    
    // Create test contour (square)
    std::vector<cv::Point2d> square = {
        {10, 10}, {10, 50}, {50, 50}, {50, 10}
    };
    
    std::cout << "  Original contour points: " << square.size() << "\n";
    
    // Apply smoothing with different parameters
    auto smoothed1 = GeometryService::LaplacianSmooth(square, 1, 0.3);
    auto smoothed5 = GeometryService::LaplacianSmooth(square, 5, 0.3);
    auto smoothed10 = GeometryService::LaplacianSmooth(square, 10, 0.5);
    
    std::cout << "  After 1 iteration (alpha=0.3): " << smoothed1.size() << " points\n";
    std::cout << "  After 5 iterations (alpha=0.3): " << smoothed5.size() << " points\n";
    std::cout << "  After 10 iterations (alpha=0.5): " << smoothed10.size() << " points\n";
    
    // Visualize
    cv::Mat viz = cv::Mat::zeros(60, 60, CV_8UC3);
    viz.setTo(cv::Scalar(255, 255, 255));
    
    // Draw original in red
    for (size_t i = 0; i < square.size(); i++) {
        cv::Point p1(square[i].y, square[i].x);
        cv::Point p2(square[(i+1)%square.size()].y, square[(i+1)%square.size()].x);
        cv::line(viz, p1, p2, cv::Scalar(0, 0, 255), 2);
    }
    
    // Draw smoothed in blue
    for (size_t i = 0; i < smoothed10.size(); i++) {
        cv::Point p1(smoothed10[i].y, smoothed10[i].x);
        cv::Point p2(smoothed10[(i+1)%smoothed10.size()].y, smoothed10[(i+1)%smoothed10.size()].x);
        cv::line(viz, p1, p2, cv::Scalar(255, 0, 0), 1);
    }
    
    cv::imwrite("test_smoothing.png", viz);
    std::cout << "  Saved: test_smoothing.png (red=original, blue=smoothed)\n";
}

void TestInflation() {
    std::cout << "\n=== Testing Contour Inflation ===\n";
    
    // Create test contour (circle)
    std::vector<cv::Point2d> circle;
    int numPoints = 32;
    double radius = 20.0;
    cv::Point2d center(30, 30);
    
    for (int i = 0; i < numPoints; i++) {
        double angle = 2.0 * CV_PI * i / numPoints;
        circle.emplace_back(
            center.x + radius * std::sin(angle),
            center.y + radius * std::cos(angle)
        );
    }
    
    std::cout << "  Original circle radius: " << radius << "\n";
    
    // Test different inflation amounts
    auto inflated5 = GeometryService::InflateContour(circle, 5.0, 1.0);
    auto inflated10 = GeometryService::InflateContour(circle, 10.0, 1.0);
    auto inflatedExp = GeometryService::InflateContour(circle, 5.0, 2.0);
    
    std::cout << "  Inflated by 5.0 (uniform)\n";
    std::cout << "  Inflated by 10.0 (uniform)\n";
    std::cout << "  Inflated by 5.0 (far point factor 2.0)\n";
    
    // Visualize
    cv::Mat viz = cv::Mat::zeros(80, 80, CV_8UC3);
    viz.setTo(cv::Scalar(255, 255, 255));
    
    auto drawContour = [&](const std::vector<cv::Point2d>& cnt, cv::Scalar color) {
        for (size_t i = 0; i < cnt.size(); i++) {
            cv::Point p1(cnt[i].y, cnt[i].x);
            cv::Point p2(cnt[(i+1)%cnt.size()].y, cnt[(i+1)%cnt.size()].x);
            cv::line(viz, p1, p2, color, 1);
        }
    };
    
    drawContour(circle, cv::Scalar(0, 0, 255));        // Red = original
    drawContour(inflated5, cv::Scalar(0, 255, 0));     // Green = +5
    drawContour(inflated10, cv::Scalar(255, 0, 0));    // Blue = +10
    drawContour(inflatedExp, cv::Scalar(255, 255, 0)); // Cyan = exponential
    
    cv::imwrite("test_inflation.png", viz);
    std::cout << "  Saved: test_inflation.png (red=original, green=+5, blue=+10, cyan=exp)\n";
}

void TestFullPipeline(const cv::Mat& image) {
    std::cout << "\n=== Testing Full Pipeline (Foundation) ===\n";
    
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // 1. Segmentation
    std::cout << "  [1/5] Segmentation...\n";
    auto segStart = std::chrono::high_resolution_clock::now();
    cv::Mat labels = SegmentationService::NoiseWatershed(image, 60.0, 2.0, 0.001);
    auto segEnd = std::chrono::high_resolution_clock::now();
    auto segTime = std::chrono::duration_cast<std::chrono::milliseconds>(segEnd - segStart);
    
    // 2. Quantization
    std::cout << "  [2/5] Quantization...\n";
    auto quantStart = std::chrono::high_resolution_clock::now();
    auto [quantized, centers] = ContourService::QuantizeColors(image, 8);
    auto quantEnd = std::chrono::high_resolution_clock::now();
    auto quantTime = std::chrono::duration_cast<std::chrono::milliseconds>(quantEnd - quantStart);
    
    // 3. Contour extraction
    std::cout << "  [3/5] Contour extraction...\n";
    auto contourStart = std::chrono::high_resolution_clock::now();
    auto contours = ContourService::ExtractContours(labels, quantized);
    auto contourEnd = std::chrono::high_resolution_clock::now();
    auto contourTime = std::chrono::duration_cast<std::chrono::milliseconds>(contourEnd - contourStart);
    
    // 4. Smoothing
    std::cout << "  [4/5] Smoothing...\n";
    auto smoothStart = std::chrono::high_resolution_clock::now();
    std::vector<ContourData> smoothedContours;
    for (auto& contourData : contours) {
        ContourData smoothed = contourData;
        smoothed.points = GeometryService::LaplacianSmooth(contourData.points, 3, 0.3);
        smoothedContours.push_back(smoothed);
    }
    auto smoothEnd = std::chrono::high_resolution_clock::now();
    auto smoothTime = std::chrono::duration_cast<std::chrono::milliseconds>(smoothEnd - smoothStart);
    
    // 5. Inflation
    std::cout << "  [5/5] Inflation...\n";
    auto inflateStart = std::chrono::high_resolution_clock::now();
    std::vector<ContourData> inflatedContours;
    for (auto& contourData : smoothedContours) {
        ContourData inflated = contourData;
        inflated.points = GeometryService::InflateContour(contourData.points, 2.0, 1.0);
        inflatedContours.push_back(inflated);
    }
    auto inflateEnd = std::chrono::high_resolution_clock::now();
    auto inflateTime = std::chrono::duration_cast<std::chrono::milliseconds>(inflateEnd - inflateStart);
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);
    
    // Report
    std::cout << "\n  Pipeline Results:\n";
    std::cout << "  ----------------\n";
    std::cout << "  Final contours: " << inflatedContours.size() << "\n";
    std::cout << "  Segmentation: " << segTime.count() << " ms\n";
    std::cout << "  Quantization: " << quantTime.count() << " ms\n";
    std::cout << "  Contour extraction: " << contourTime.count() << " ms\n";
    std::cout << "  Smoothing: " << smoothTime.count() << " ms\n";
    std::cout << "  Inflation: " << inflateTime.count() << " ms\n";
    std::cout << "  TOTAL: " << totalTime.count() << " ms\n";
    
    // Visualize final result
    cv::Mat visualization = cv::Mat::zeros(image.size(), CV_8UC3);
    visualization.setTo(cv::Scalar(255, 255, 255));
    
    for (const auto& contourData : inflatedContours) {
        std::vector<cv::Point> cvPoints;
        for (const auto& pt : contourData.points) {
            cvPoints.emplace_back(static_cast<int>(pt.y), static_cast<int>(pt.x));
        }
        
        cv::Scalar color(contourData.color[0], contourData.color[1], contourData.color[2]);
        cv::fillPoly(visualization, std::vector<std::vector<cv::Point>>{cvPoints}, color);
    }
    
    cv::imwrite("test_pipeline_result.png", visualization);
    std::cout << "  Saved: test_pipeline_result.png\n";
}

void TestCompletePipeline(const cv::Mat& image) {
    std::cout << "\n=== Testing Complete Vectorization Pipeline ===\n";
    
    // Setup parameters
    SegmentationParams segParams;
    segParams.noiseScale = 60.0;
    segParams.blurSigma = 2.0;
    segParams.compactness = 0.001;
    segParams.referenceCropSize = cv::Size(512, 512);
    segParams.scalingMethod = ScalingMethod::SQRT_AREA;
    segParams.segmentationMultiplier = 1.0;
    
    QuantizationParams quantParams;
    quantParams.maxColors = 8;
    
    BridgingParams bridgingParams;
    bridgingParams.proximityThreshold = 20.0;
    bridgingParams.colorTolerance = 30.0;
    bridgingParams.bridgeDistance = 3.0;
    bridgingParams.falloffRadius = 5;
    bridgingParams.maxCurvature = 45.0;
    
    SmoothingParams smoothParams;
    smoothParams.smoothingIterations = 5;
    smoothParams.smoothingAlpha = 0.5;
    
    InflationParams inflationParams;
    inflationParams.inflationAmount = 2.0;
    inflationParams.farPointFactor = 1.5;
    
    DropletParams dropletParams;
    // Not used yet
    
    SVGParams svgParams;
    svgParams.stackingOrder = StackingOrder::AREA;
    svgParams.simplifyTolerance = 1.0;
    svgParams.quantizeCoordinates = true;
    
    ModifierFlags modifiers;
    modifiers.enableBridging = true;
    modifiers.enableSmoothing = true;
    modifiers.enableInflation = true;
    
    // Execute pipeline
    auto start = std::chrono::high_resolution_clock::now();
    auto result = VectorizationPipeline::Execute(
        image, segParams, quantParams, bridgingParams,
        smoothParams, inflationParams, dropletParams,
        svgParams, modifiers
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "  Total contours: " << result.contours.size() << "\n";
    std::cout << "  Z-order indices: " << result.zOrderIndices.size() << "\n";
    std::cout << "  Total time: " << duration << " ms\n";
    std::cout << "  Stage timings:\n";
    for (const auto& [stage, time] : result.timings) {
        std::cout << "    " << stage << ": " << time << " ms\n";
    }
    
    // Export to SVG
    bool success = VectorizationPipeline::ExportToSVG(result, "test_complete_pipeline.svg", svgParams);
    if (success) {
        std::cout << "  Saved: test_complete_pipeline.svg\n";
    } else {
        std::cout << "  ERROR: Failed to save SVG\n";
    }
}

int main(int argc, char** argv) {
    std::cout << "=== Organic Vectoriser C++ Foundation Tests ===\n";
    
    // Load test image
    std::string imagePath = (argc > 1) ? argv[1] : "test_input.png";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    
    if (image.empty()) {
        std::cerr << "Error: Could not load image: " << imagePath << "\n";
        std::cerr << "Usage: " << argv[0] << " [image_path]\n";
        std::cerr << "\nCreating synthetic test image...\n";
        
        // Create a simple synthetic test image
        image = cv::Mat(240, 360, CV_8UC3);
        cv::rectangle(image, cv::Rect(50, 50, 100, 100), cv::Scalar(255, 0, 0), -1);
        cv::rectangle(image, cv::Rect(200, 50, 100, 100), cv::Scalar(0, 255, 0), -1);
        cv::circle(image, cv::Point(180, 180), 50, cv::Scalar(0, 0, 255), -1);
        cv::imwrite("test_input.png", image);
        std::cout << "Created: test_input.png\n";
    }
    
    std::cout << "Image loaded: " << image.cols << "x" << image.rows << "\n";
    
    // Run tests
    try {
        TestSegmentation(image);
        TestQuantization(image);
        TestContourExtraction(image);
        TestSmoothing();
        TestInflation();
        TestFullPipeline(image);
        TestCompletePipeline(image);
        
        std::cout << "\n=== All Tests Completed Successfully ===\n";
        std::cout << "\nGenerated files:\n";
        std::cout << "  - test_segmentation_labels.png\n";
        std::cout << "  - test_quantized.png\n";
        std::cout << "  - test_contours.png\n";
        std::cout << "  - test_smoothing.png\n";
        std::cout << "  - test_inflation.png\n";
        std::cout << "  - test_pipeline_result.png\n";
        std::cout << "  - test_complete_pipeline.svg\n";
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }
}
