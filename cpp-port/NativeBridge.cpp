#include "NativeBridge.h"
#include "VectorizationPipeline.h"
#include "SVGWriter.h"
#include <vector>
#include <cstring>

using namespace Vectoriser;

namespace {
    StackingOrder ToStackingOrder(int value) {
        // Clamp to valid enum range
        if (value < static_cast<int>(StackingOrder::AREA)) {
            return StackingOrder::AREA;
        }
        if (value > static_cast<int>(StackingOrder::POSITION_CENTRE_REVERSE)) {
            return StackingOrder::POSITION_CENTRE_REVERSE;
        }
        return static_cast<StackingOrder>(value);
    }

    // Copy contour points (note: internal points are stored as (row,col) so swap to (x,y)
    void CopyContours(const std::vector<ContourData>& src, OVContour* dst) {
        for (size_t i = 0; i < src.size(); ++i) {
            const auto& contour = src[i];
            dst[i].pointCount = static_cast<int>(contour.points.size());
            dst[i].points = nullptr;
            if (!contour.points.empty()) {
                dst[i].points = new OVPoint[contour.points.size()];
                for (size_t j = 0; j < contour.points.size(); ++j) {
                    // Convert from (row, col) to (x, y)
                    dst[i].points[j].x = contour.points[j].y;
                    dst[i].points[j].y = contour.points[j].x;
                }
            }
            dst[i].color[0] = contour.color[0];
            dst[i].color[1] = contour.color[1];
            dst[i].color[2] = contour.color[2];
        }
    }

    void CopyDroplets(const std::vector<DropletDescriptor>& src, OVDroplet* dst) {
        for (size_t i = 0; i < src.size(); ++i) {
            const auto& d = src[i];
            dst[i].kind = static_cast<int>(d.type);
            dst[i].cx = d.center.x;
            dst[i].cy = d.center.y;
            dst[i].rx = d.size.x;
            dst[i].ry = d.size.y;
            dst[i].angleDegrees = d.rotation;
            dst[i].color[0] = d.color[0];
            dst[i].color[1] = d.color[1];
            dst[i].color[2] = d.color[2];
            dst[i].pointCount = static_cast<int>(d.polygon.size());
            dst[i].polygon = nullptr;
            if (!d.polygon.empty()) {
                dst[i].polygon = new OVPoint[d.polygon.size()];
                for (size_t j = 0; j < d.polygon.size(); ++j) {
                    dst[i].polygon[j].x = d.polygon[j].y;
                    dst[i].polygon[j].y = d.polygon[j].x;
                }
            }
        }
    }
}

extern "C" {

int ov_run_pipeline(const uint8_t* bgrData,
                    int width,
                    int height,
                    const OVParams* p,
                    const OVModifiers* m,
                    OVResult** outResult,
                    const char* svgOutputPath) {
    if (!bgrData || width <= 0 || height <= 0 || !p || !m || !outResult) {
        return -1;
    }

    try {
        cv::Mat bgr(height, width, CV_8UC3, const_cast<uint8_t*>(bgrData));

        // Build parameter structs
        SegmentationParams seg;
        seg.noiseScale = p->noiseScale;
        seg.blurSigma = p->blurSigma;
        seg.compactness = p->compactness;
        seg.segmentationMultiplier = p->segmentationMultiplier;
        seg.referenceCropSize = cv::Size(512, 512);
        seg.scalingMethod = ScalingMethod::SQRT_AREA;

        QuantizationParams quant;
        quant.maxColors = p->maxColors;

        BridgingParams bridge;
        bridge.proximityThreshold = p->proximityThreshold;
        bridge.colorTolerance = p->colorTolerance;
        bridge.bridgeDistance = p->bridgeDistance;
        bridge.falloffRadius = p->falloffRadius;
        bridge.maxCurvature = p->maxCurvatureDegrees;

        SmoothingParams smooth;
        smooth.iterations = p->smoothIterations;
        smooth.alpha = p->smoothAlpha;

        InflationParams inflate;
        inflate.inflationAmount = p->blobInflationAmount;
        inflate.farPointFactor = p->farPointInflationFactor;
        inflate.proportionalToStacking = p->inflationProportionalToStacking != 0;

        DropletParams droplet;
        droplet.style = (p->dropletOrganicDensity > 0) ? DropletStyle::ORGANIC : DropletStyle::PAINTERLY;
        droplet.painterly.density = p->dropletDensity;
        droplet.painterly.minDistance = p->dropletMinDistance;
        droplet.painterly.maxDistance = p->dropletMaxDistance;
        droplet.painterly.sizeMean = p->dropletSizeMean;
        droplet.painterly.sizeStd = p->dropletSizeStd;
        droplet.painterly.spreadAngle = p->dropletSpreadDegrees;
        droplet.painterly.globalRotation = p->dropletGlobalRotation;
        droplet.painterly.useSVGPrimitives = p->painterlyUseSvgEllipses != 0;
        droplet.painterly.rectHorizontal = p->painterlyRectHorizontal != 0;
        droplet.painterly.primitiveType = p->painterlyUseSvgEllipses
            ? (p->painterlyRectHorizontal ? PainterlyPrimitive::RECT : PainterlyPrimitive::ELLIPSE)
            : PainterlyPrimitive::POLYGON;

        droplet.organic.minBrightness = p->dropletOrganicMinBrightness;
        droplet.organic.density = p->dropletOrganicDensity;
        droplet.organic.strength = p->dropletOrganicStrength;
        droplet.organic.jitter = p->dropletOrganicJitter;
        droplet.organic.elongation = p->dropletOrganicElongation;
        droplet.organic.percentPerBlob = p->dropletOrganicPercentPerBlob;

        SVGParams svgParams;
        svgParams.simplifyTolerance = p->simplifyTolerance;
        svgParams.quantizeCoordinates = true;
        svgParams.stackingOrder = ToStackingOrder(p->stackingOrder);

        ModifierFlags mods;
        mods.enableQuantization = m->colorQuantization != 0;
        mods.enableBridging = m->bridging != 0;
        mods.enableSmoothing = m->smoothing != 0;
        mods.enableInflation = m->inflation != 0;

        auto result = VectorizationPipeline::Execute(
            bgr, seg, quant, bridge, smooth, inflate, droplet, svgParams, mods);

        // Allocate native result
        OVResult* native = new OVResult();
        native->width = bgr.cols;
        native->height = bgr.rows;
        native->contourCount = static_cast<int>(result.contours.size());
        native->dropletCount = static_cast<int>(result.droplets.size());
        native->contours = nullptr;
        native->droplets = nullptr;

        if (!result.contours.empty()) {
            native->contours = new OVContour[result.contours.size()];
            std::memset(native->contours, 0, sizeof(OVContour) * result.contours.size());
            CopyContours(result.contours, native->contours);
        }

        if (!result.droplets.empty()) {
            native->droplets = new OVDroplet[result.droplets.size()];
            std::memset(native->droplets, 0, sizeof(OVDroplet) * result.droplets.size());
            CopyDroplets(result.droplets, native->droplets);
        }

        // Optional SVG export
        if (svgOutputPath && svgOutputPath[0] != '\0') {
            VectorizationPipeline::ExportToSVG(result, svgOutputPath, svgParams);
        }

        *outResult = native;
        return 0;
    }
    catch (...) {
        return -2;
    }
}

void ov_free_result(OVResult* result) {
    if (!result) return;
    if (result->contours) {
        for (int i = 0; i < result->contourCount; ++i) {
            delete[] result->contours[i].points;
        }
        delete[] result->contours;
    }
    if (result->droplets) {
        for (int i = 0; i < result->dropletCount; ++i) {
            delete[] result->droplets[i].polygon;
        }
        delete[] result->droplets;
    }
    delete result;
}

} // extern "C"
