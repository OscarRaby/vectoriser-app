#include "SVGWriter.h"
#include <sstream>
#include <iomanip>
#include <cmath>

namespace Vectoriser {

std::string SVGWriter::FormatRGBColor(const cv::Vec3b& color) {
    std::ostringstream oss;
    oss << "rgb(" << static_cast<int>(color[0]) << ","
        << static_cast<int>(color[1]) << ","
        << static_cast<int>(color[2]) << ")";
    return oss.str();
}

void SVGWriter::WriteSVGHeader(std::ofstream& file, cv::Size imageSize) {
    file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    file << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
         << "width=\"" << imageSize.width << "\" "
         << "height=\"" << imageSize.height << "\" "
         << "viewBox=\"0 0 " << imageSize.width << " " << imageSize.height << "\">\n";
    file << "  <style>path { stroke: none; stroke-width: 0; }</style>\n";
}

void SVGWriter::WriteSVGFooter(std::ofstream& file) {
    file << "</svg>\n";
}

std::vector<cv::Point2d> SVGWriter::SimplifyContour(
    const std::vector<cv::Point2d>& contour,
    double tolerance
) {
    if (tolerance <= 0.0 || contour.size() < 3) {
        return contour;
    }
    
    // Use OpenCV's approxPolyDP for Douglas-Peucker simplification
    std::vector<cv::Point2f> contourFloat;
    contourFloat.reserve(contour.size());
    for (const auto& pt : contour) {
        contourFloat.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
    }
    
    std::vector<cv::Point2f> simplified;
    cv::approxPolyDP(contourFloat, simplified, tolerance, true);
    
    // Convert back to Point2d
    std::vector<cv::Point2d> result;
    result.reserve(simplified.size());
    for (const auto& pt : simplified) {
        result.emplace_back(pt.x, pt.y);
    }
    
    return result;
}

std::string SVGWriter::ContourToSVGPath(
    const std::vector<cv::Point2d>& contour,
    double simplifyTolerance,
    bool quantize
) {
    if (contour.empty()) return "";
    
    // Simplify if requested
    auto points = (simplifyTolerance > 0.0) ? SimplifyContour(contour, simplifyTolerance) : contour;
    if (points.empty()) return "";
    
    std::ostringstream path;
    path << "M ";
    
    for (size_t i = 0; i < points.size(); i++) {
        // Convert from (row, col) to SVG (x, y)
        double x = points[i].y;  // col -> x
        double y = points[i].x;  // row -> y
        
        if (quantize) {
            path << static_cast<int>(std::round(x)) << ","
                 << static_cast<int>(std::round(y));
        } else {
            path << std::fixed << std::setprecision(1) << x << "," << y;
        }
        
        if (i == 0) {
            path << " L ";
        } else if (i < points.size() - 1) {
            path << " ";
        }
    }
    
    path << " Z";
    return path.str();
}

void SVGWriter::WriteContourPath(
    std::ofstream& file,
    const std::vector<cv::Point2d>& contour,
    const cv::Vec3b& color,
    const SVGParams& params
) {
    std::string pathData = ContourToSVGPath(contour, params.simplifyTolerance, params.quantizeCoordinates);
    if (pathData.empty()) return;
    
    file << "  <path d=\"" << pathData << "\" "
         << "fill=\"" << FormatRGBColor(color) << "\" />\n";
}

void SVGWriter::WriteDroplet(
    std::ofstream& file,
    const DropletDescriptor& droplet,
    const SVGParams& params
) {
    std::string colorStr = FormatRGBColor(droplet.color);
    
    switch (droplet.type) {
        case DropletDescriptor::Type::ELLIPSE:
            file << "  <ellipse "
                 << "cx=\"" << droplet.center.x << "\" "
                 << "cy=\"" << droplet.center.y << "\" "
                 << "rx=\"" << droplet.size.x << "\" "
                 << "ry=\"" << droplet.size.y << "\" "
                 << "fill=\"" << colorStr << "\" "
                 << "transform=\"rotate(" << droplet.rotation << " "
                 << droplet.center.x << " " << droplet.center.y << ")\" />\n";
            break;
            
        case DropletDescriptor::Type::RECT:
            {
                double x = droplet.center.x - droplet.size.x / 2.0;
                double y = droplet.center.y - droplet.size.y / 2.0;
                file << "  <rect "
                     << "x=\"" << x << "\" "
                     << "y=\"" << y << "\" "
                     << "width=\"" << droplet.size.x << "\" "
                     << "height=\"" << droplet.size.y << "\" "
                     << "fill=\"" << colorStr << "\" "
                     << "transform=\"rotate(" << droplet.rotation << " "
                     << droplet.center.x << " " << droplet.center.y << ")\" />\n";
            }
            break;
            
        case DropletDescriptor::Type::POLYGON:
            if (!droplet.polygon.empty()) {
                std::string pathData = ContourToSVGPath(droplet.polygon, params.simplifyTolerance, params.quantizeCoordinates);
                if (!pathData.empty()) {
                    file << "  <path d=\"" << pathData << "\" "
                         << "fill=\"" << colorStr << "\" />\n";
                }
            }
            break;
    }
}

bool SVGWriter::WriteSVG(
    const std::string& filename,
    const std::vector<ContourData>& contours,
    const std::vector<DropletDescriptor>& droplets,
    const std::vector<size_t>& zOrderIndices,
    cv::Size imageSize,
    const SVGParams& params
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    WriteSVGHeader(file, imageSize);
    
    // Write contours in z-order
    for (size_t idx : zOrderIndices) {
        if (idx < contours.size()) {
            WriteContourPath(file, contours[idx].points, contours[idx].color, params);
        }
    }
    
    // Write droplets
    for (const auto& droplet : droplets) {
        WriteDroplet(file, droplet, params);
    }
    
    WriteSVGFooter(file);
    file.close();
    
    return true;
}

} // namespace Vectoriser
