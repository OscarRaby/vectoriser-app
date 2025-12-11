#include "VectorizationPipeline.h"
#include "ParameterScaler.h"
#include "SegmentationService.h"
#include "ContourService.h"
#include "BridgingService.h"
#include "GeometryService.h"
#include "SortingService.h"
#include "SVGWriter.h"
#include <chrono>

namespace Vectoriser {

std::vector<cv::Point2d> VectorizationPipeline::CalculateCentroids(
    const std::vector<std::vector<cv::Point2d>>& contours
) {
    std::vector<cv::Point2d> centroids;
    centroids.reserve(contours.size());
    
    for (const auto& contour : contours) {
        cv::Point2d centroid(0.0, 0.0);
        if (!contour.empty()) {
            for (const auto& pt : contour) {
                centroid += pt;
            }
            centroid /= static_cast<double>(contour.size());
        }
        centroids.push_back(centroid);
    }
    
    return centroids;
}

std::vector<cv::Vec3b> VectorizationPipeline::ExtractColors(
    const std::vector<ContourData>& contours
) {
    std::vector<cv::Vec3b> colors;
    colors.reserve(contours.size());
    for (const auto& contour : contours) {
        colors.push_back(contour.color);
    }
    return colors;
}

std::vector<ContourData> VectorizationPipeline::ContoursToContourData(
    const std::vector<std::vector<cv::Point2d>>& contours,
    const std::vector<cv::Vec3b>& colors
) {
    std::vector<ContourData> result;
    result.reserve(contours.size());
    
    for (size_t i = 0; i < contours.size(); i++) {
        ContourData data;
        data.points = contours[i];
        data.color = (i < colors.size()) ? colors[i] : cv::Vec3b(0, 0, 0);
        data.area = ContourService::CalculateArea(contours[i]);
        data.luminance = ContourService::CalculateLuminance(data.color);
        
        // Calculate centroid
        if (!contours[i].empty()) {
            cv::Point2d centroid(0.0, 0.0);
            for (const auto& pt : contours[i]) {
                centroid += pt;
            }
            data.centroid = centroid / static_cast<double>(contours[i].size());
        }
        
        result.push_back(data);
    }
    
    return result;
}

void VectorizationPipeline::ExecuteSegmentation(
    const cv::Mat& input,
    const SegmentationParams& params,
    cv::Mat& labels,
    int& numRegions,
    double& timingMs
) {
    auto start = std::chrono::high_resolution_clock::now();
    labels = SegmentationService::NoiseWatershed(input, params.noiseScale, params.blurSigma, params.compactness);
    
    // Count unique regions
    double minVal, maxVal;
    cv::minMaxLoc(labels, &minVal, &maxVal);
    numRegions = static_cast<int>(maxVal);
    
    auto end = std::chrono::high_resolution_clock::now();
    timingMs = std::chrono::duration<double, std::milli>(end - start).count();
}

void VectorizationPipeline::ExecuteQuantization(
    const cv::Mat& input,
    const QuantizationParams& params,
    cv::Mat& quantized,
    double& timingMs
) {
    auto start = std::chrono::high_resolution_clock::now();
    auto [quantizedMat, centers] = ContourService::QuantizeColors(input, params.maxColors);
    quantized = quantizedMat;
    auto end = std::chrono::high_resolution_clock::now();
    timingMs = std::chrono::duration<double, std::milli>(end - start).count();
}

void VectorizationPipeline::ExecuteContourExtraction(
    const cv::Mat& labels,
    const cv::Mat& quantized,
    std::vector<ContourData>& contours,
    double& timingMs
) {
    auto start = std::chrono::high_resolution_clock::now();
    contours = ContourService::ExtractContours(labels, quantized);
    auto end = std::chrono::high_resolution_clock::now();
    timingMs = std::chrono::duration<double, std::milli>(end - start).count();
}

void VectorizationPipeline::ExecuteBridging(
    std::vector<ContourData>& contours,
    const BridgingParams& params,
    double& timingMs
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Extract contours, centroids, and colors
    std::vector<std::vector<cv::Point2d>> points;
    std::vector<cv::Point2d> centroids;
    std::vector<cv::Vec3b> colors;
    
    points.reserve(contours.size());
    centroids.reserve(contours.size());
    colors.reserve(contours.size());
    
    for (const auto& c : contours) {
        points.push_back(c.points);
        centroids.push_back(c.centroid);
        colors.push_back(c.color);
    }
    
    // Apply bridging
    auto bridged = BridgingService::ImprovedBridgeContours(points, centroids, colors, params);
    
    // Update contours with bridged points
    for (size_t i = 0; i < contours.size() && i < bridged.size(); i++) {
        contours[i].points = bridged[i];
        
        // Recalculate centroid
        if (!bridged[i].empty()) {
            cv::Point2d centroid(0.0, 0.0);
            for (const auto& pt : bridged[i]) {
                centroid += pt;
            }
            contours[i].centroid = centroid / static_cast<double>(bridged[i].size());
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    timingMs = std::chrono::duration<double, std::milli>(end - start).count();
}

void VectorizationPipeline::ExecuteSmoothing(
    std::vector<ContourData>& contours,
    const SmoothingParams& params,
    double& timingMs
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (auto& contour : contours) {
        contour.points = GeometryService::LaplacianSmooth(
            contour.points,
            params.smoothingIterations,
            params.smoothingAlpha
        );
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    timingMs = std::chrono::duration<double, std::milli>(end - start).count();
}

void VectorizationPipeline::ExecuteInflation(
    std::vector<ContourData>& contours,
    const InflationParams& params,
    double& timingMs
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (auto& contour : contours) {
        contour.points = GeometryService::InflateContour(
            contour.points,
            params.inflationAmount,
            params.farPointFactor
        );
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    timingMs = std::chrono::duration<double, std::milli>(end - start).count();
}

void VectorizationPipeline::ExecuteSorting(
    const std::vector<ContourData>& contours,
    StackingOrder order,
    cv::Size imageSize,
    std::vector<size_t>& zOrderIndices,
    double& timingMs
) {
    auto start = std::chrono::high_resolution_clock::now();
    zOrderIndices = SortingService::SortContoursByOrder(contours, order, imageSize);
    auto end = std::chrono::high_resolution_clock::now();
    timingMs = std::chrono::duration<double, std::milli>(end - start).count();
}

PipelineResult VectorizationPipeline::Execute(
    const cv::Mat& inputImage,
    const SegmentationParams& segParams,
    const QuantizationParams& quantParams,
    const BridgingParams& bridgingParams,
    const SmoothingParams& smoothParams,
    const InflationParams& inflationParams,
    const DropletParams& dropletParams,
    const SVGParams& svgParams,
    const ModifierFlags& modifiers
) {
    PipelineResult result;
    result.imageSize = inputImage.size();
    
    // Ensure input is RGB
    cv::Mat rgb = inputImage;
    if (inputImage.channels() == 1) {
        cv::cvtColor(inputImage, rgb, cv::COLOR_GRAY2RGB);
    } else if (inputImage.channels() == 4) {
        cv::cvtColor(inputImage, rgb, cv::COLOR_RGBA2RGB);
    }
    
    // Scale parameters based on image size
    SegmentationParams scaledSegParams = ParameterScaler::ScaleParameters(segParams, rgb.size());
    
    // Stage 1: Segmentation
    cv::Mat labels;
    int numRegions = 0;
    ExecuteSegmentation(rgb, scaledSegParams, labels, numRegions, result.timings["segmentation"]);
    
    // Stage 2: Quantization
    cv::Mat quantized;
    ExecuteQuantization(rgb, quantParams, quantized, result.timings["quantization"]);
    
    // Stage 3: Contour Extraction
    std::vector<ContourData> contours;
    ExecuteContourExtraction(labels, quantized, contours, result.timings["contour_extraction"]);
    
    // Stage 4: Bridging (optional)
    if (modifiers.enableBridging) {
        ExecuteBridging(contours, bridgingParams, result.timings["bridging"]);
    }
    
    // Stage 5: Smoothing (optional)
    if (modifiers.enableSmoothing) {
        ExecuteSmoothing(contours, smoothParams, result.timings["smoothing"]);
    }
    
    // Stage 6: Inflation (optional)
    if (modifiers.enableInflation) {
        ExecuteInflation(contours, inflationParams, result.timings["inflation"]);
    }
    
    // Stage 7: Sorting
    std::vector<size_t> zOrderIndices;
    ExecuteSorting(contours, svgParams.stackingOrder, rgb.size(), zOrderIndices, result.timings["sorting"]);
    
    // Store results
    result.contours = std::move(contours);
    result.zOrderIndices = std::move(zOrderIndices);
    
    // Calculate total time
    result.totalTimeMs = 0.0;
    for (const auto& timing : result.timings) {
        result.totalTimeMs += timing.second;
    }
    
    return result;
}

bool VectorizationPipeline::ExportToSVG(
    const PipelineResult& result,
    const std::string& filename,
    const SVGParams& params
) {
    return SVGWriter::WriteSVG(
        filename,
        result.contours,
        result.droplets,
        result.zOrderIndices,
        result.imageSize,
        params
    );
}

} // namespace Vectoriser
