#pragma once

#include "VectorizationParameters.h"
#include <opencv2/opencv.hpp>
#include <string>

namespace Vectoriser {

class VectorizationPipeline {
public:
    // Run the complete vectorization pipeline
    // Matches Python: run_vectoriser_compute()
    static PipelineResult Execute(
        const cv::Mat& inputImage,
        const SegmentationParams& segParams,
        const QuantizationParams& quantParams,
        const BridgingParams& bridgingParams,
        const SmoothingParams& smoothParams,
        const InflationParams& inflationParams,
        const DropletParams& dropletParams,
        const SVGParams& svgParams,
        const ModifierFlags& modifiers
    );
    
    // Export result to SVG file
    static bool ExportToSVG(
        const PipelineResult& result,
        const std::string& filename,
        const SVGParams& params
    );
    
private:
    // Calculate centroids for contours
    static std::vector<cv::Point2d> CalculateCentroids(
        const std::vector<std::vector<cv::Point2d>>& contours
    );
    
    // Extract colors from contour data
    static std::vector<cv::Vec3b> ExtractColors(
        const std::vector<ContourData>& contours
    );
    
    // Convert Point2d contours to ContourData
    static std::vector<ContourData> ContoursToContourData(
        const std::vector<std::vector<cv::Point2d>>& contours,
        const std::vector<cv::Vec3b>& colors
    );
    
    // Stage execution helpers
    static void ExecuteSegmentation(
        const cv::Mat& input,
        const SegmentationParams& params,
        cv::Mat& labels,
        int& numRegions,
        double& timingMs
    );
    
    static void ExecuteQuantization(
        const cv::Mat& input,
        const QuantizationParams& params,
        cv::Mat& quantized,
        double& timingMs
    );
    
    static void ExecuteContourExtraction(
        const cv::Mat& labels,
        const cv::Mat& quantized,
        std::vector<ContourData>& contours,
        double& timingMs
    );
    
    static void ExecuteBridging(
        std::vector<ContourData>& contours,
        const BridgingParams& params,
        double& timingMs
    );
    
    static void ExecuteSmoothing(
        std::vector<ContourData>& contours,
        const SmoothingParams& params,
        double& timingMs
    );
    
    static void ExecuteInflation(
        std::vector<ContourData>& contours,
        const InflationParams& params,
        double& timingMs
    );
    
    static void ExecuteSorting(
        const std::vector<ContourData>& contours,
        StackingOrder order,
        cv::Size imageSize,
        std::vector<size_t>& zOrderIndices,
        double& timingMs
    );
};

} // namespace Vectoriser
