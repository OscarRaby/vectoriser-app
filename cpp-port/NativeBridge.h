#pragma once

#include "VectorizationParameters.h"
#include <cstdint>

#ifdef _WIN32
#define OV_API __declspec(dllexport)
#else
#define OV_API
#endif

extern "C" {

// Parameter struct matching C# ParameterSet
struct OVParams {
    double noiseScale;
    double blurSigma;
    double compactness;
    int maxColors;
    double bridgeDistance;
    double colorTolerance;
    double proximityThreshold;
    int falloffRadius;
    double maxCurvatureDegrees;
    int smoothIterations;
    double smoothAlpha;
    double blobInflationAmount;
    double farPointInflationFactor;
    int inflationProportionalToStacking; // bool
    int stackingOrder;                   // matches Vectoriser::StackingOrder enum
    double segmentationMultiplier;
    int dropletDensity;
    double dropletMinDistance;
    double dropletMaxDistance;
    double dropletSizeMean;
    double dropletSizeStd;
    double dropletSpreadDegrees;
    double dropletOrganicMinBrightness;
    int dropletOrganicDensity;
    double dropletOrganicStrength;
    double dropletOrganicJitter;
    double dropletOrganicElongation;
    double dropletOrganicPercentPerBlob;
    int painterlyUseSvgEllipses; // bool
    int painterlyRectHorizontal; // bool
    double dropletGlobalRotation;
    double simplifyTolerance;
};

// Modifier flags matching C# ModifierFlags
struct OVModifiers {
    int colorQuantization; // bool
    int bridging;          // bool
    int smoothing;         // bool
    int inflation;         // bool
    int enableVectorPreview; // bool (unused by native pipeline)
};

struct OVPoint {
    double x;
    double y;
};

struct OVContour {
    OVPoint* points;
    int pointCount;
    uint8_t color[3];
};

struct OVDroplet {
    int kind; // 0=Polygon,1=Ellipse,2=Rect
    OVPoint* polygon;
    int pointCount;
    double cx;
    double cy;
    double rx;
    double ry;
    double angleDegrees;
    uint8_t color[3];
};

struct OVResult {
    OVContour* contours;
    int contourCount;
    OVDroplet* droplets;
    int dropletCount;
    int width;
    int height;
};

// Run full pipeline. Input is BGR pixel buffer (width*height*3).
// Returns 0 on success. Caller must free OVResult with ov_free_result.
OV_API int ov_run_pipeline(const uint8_t* bgrData,
                           int width,
                           int height,
                           const OVParams* params,
                           const OVModifiers* modifiers,
                           OVResult** outResult,
                           const char* svgOutputPath);

// Free result allocated by ov_run_pipeline
OV_API void ov_free_result(OVResult* result);

} // extern "C"
