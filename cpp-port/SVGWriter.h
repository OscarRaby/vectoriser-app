#pragma once

#include "VectorizationParameters.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>

namespace Vectoriser {

class SVGWriter {
public:
    // Write contours and droplets to SVG file
    static bool WriteSVG(
        const std::string& filename,
        const std::vector<ContourData>& contours,
        const std::vector<DropletDescriptor>& droplets,
        const std::vector<size_t>& zOrderIndices,
        cv::Size imageSize,
        const SVGParams& params
    );
    
private:
    // Convert contour to SVG path string
    // Matches Python: contour_to_svg_path()
    static std::string ContourToSVGPath(
        const std::vector<cv::Point2d>& contour,
        double simplifyTolerance,
        bool quantize
    );
    
    // Simplify contour using Douglas-Peucker algorithm
    static std::vector<cv::Point2d> SimplifyContour(
        const std::vector<cv::Point2d>& contour,
        double tolerance
    );
    
    // Format RGB color as "rgb(r,g,b)"
    static std::string FormatRGBColor(const cv::Vec3b& color);
    
    // Write SVG header
    static void WriteSVGHeader(std::ofstream& file, cv::Size imageSize);
    
    // Write SVG footer
    static void WriteSVGFooter(std::ofstream& file);
    
    // Write a single contour as path element
    static void WriteContourPath(
        std::ofstream& file,
        const std::vector<cv::Point2d>& contour,
        const cv::Vec3b& color,
        const SVGParams& params
    );
    
    // Write a droplet element
    static void WriteDroplet(
        std::ofstream& file,
        const DropletDescriptor& droplet,
        const SVGParams& params
    );
};

} // namespace Vectoriser
