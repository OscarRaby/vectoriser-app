using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Media;
using Accord.Collections;
using OpenCvSharp;

namespace OrganicVectoriser.Services;

/// <summary>
/// Service for contour processing operations: bridging, inflation, smoothing, simplification.
/// </summary>
public static class ContourService
{
    /// <summary>
    /// Bridge contours by displacing points toward nearby similarly-colored contours.
    /// Uses KD-tree for proximity queries, Lab color space for color distance, and curvature checks.
    /// </summary>
    public static List<List<Point>> BridgeContours(
        List<List<Point>> contours,
        List<Point2f> centroids,
        List<(byte, byte, byte)> colors,
        double bridgeDistance,
        double colorTolerance,
        double proximityThreshold,
        int falloffRadius,
        double maxCurvatureRadians)
    {
        if (contours.Count != centroids.Count || contours.Count != colors.Count)
            throw new ArgumentException("Contours, centroids, and colors must have matching counts.");

        // Convert RGB to Lab color space for perceptual color distance
        var labColors = colors.Select(c => RgbToLab(c.Item1, c.Item2, c.Item3)).ToList();

        // Build KD-tree for centroid proximity queries
        var centroidPoints = centroids.Select(c => new[] { (double)c.X, (double)c.Y }).ToArray();
        var kdTree = KDTree.FromData(centroidPoints, Enumerable.Range(0, centroids.Count).ToArray());

        var bridgedContours = new List<List<Point>>();

        for (int i = 0; i < contours.Count; i++)
        {
            var currentContour = contours[i].Select(p => new Point(p.X, p.Y)).ToList();
            var currentColor = labColors[i];
            var currentCentroid = centroids[i];
            int n = currentContour.Count;

            // Find nearby contours within proximity threshold
            var neighbors = kdTree.Nearest(new[] { (double)currentCentroid.X, (double)currentCentroid.Y }, proximityThreshold);

            foreach (var neighbor in neighbors)
            {
                int j = neighbor.Node.Value;
                if (j == i) continue;

                // Check color similarity in Lab space
                var colorDiff = LabDistance(currentColor, labColors[j]);
                if (colorDiff >= colorTolerance) continue;

                var targetCentroid = centroids[j];
                var targetContour = contours[j];

                // Find closest point on current contour to target centroid
                var distances = currentContour.Select(p => Distance(p, targetCentroid)).ToArray();
                int closestIdx = Array.IndexOf(distances, distances.Min());

                // Find closest point on target contour to the closest point on current
                var closestOnTarget = targetContour
                    .OrderBy(p => Distance(p, currentContour[closestIdx]))
                    .First();

                // Direction vector
                var direction = new Point2f(
                    closestOnTarget.X - currentContour[closestIdx].X,
                    closestOnTarget.Y - currentContour[closestIdx].Y
                );
                var norm = Math.Sqrt(direction.X * direction.X + direction.Y * direction.Y);
                if (norm < 1e-6) continue;
                direction = new Point2f(direction.X / (float)norm, direction.Y / (float)norm);

                // Apply displacement with falloff to neighboring points
                for (int offset = -falloffRadius; offset <= falloffRadius; offset++)
                {
                    int neighborIdx = (closestIdx + offset + n) % n;
                    int prevIdx = (neighborIdx - 1 + n) % n;
                    int nextIdx = (neighborIdx + 1) % n;

                    // Check curvature (angle between consecutive segments)
                    var v1 = Subtract(currentContour[neighborIdx], currentContour[prevIdx]);
                    var v2 = Subtract(currentContour[nextIdx], currentContour[neighborIdx]);
                    var normV1 = Math.Sqrt(v1.X * v1.X + v1.Y * v1.Y);
                    var normV2 = Math.Sqrt(v2.X * v2.X + v2.Y * v2.Y);
                    if (normV1 < 1e-6 || normV2 < 1e-6) continue;

                    var cosAngle = (v1.X * v2.X + v1.Y * v2.Y) / (normV1 * normV2);
                    var angle = Math.Acos(Math.Clamp(cosAngle, -1.0, 1.0));
                    if (angle > maxCurvatureRadians) continue;

                    // Cosine falloff weight
                    var weight = 0.5 * (1 + Math.Cos(Math.PI * offset / falloffRadius));
                    var displacement = new Point2f(
                        direction.X * (float)(bridgeDistance * weight),
                        direction.Y * (float)(bridgeDistance * weight)
                    );

                    currentContour[neighborIdx] = new Point(
                        (int)(currentContour[neighborIdx].X + displacement.X),
                        (int)(currentContour[neighborIdx].Y + displacement.Y)
                    );
                }
            }

            bridgedContours.Add(currentContour);
        }

        return bridgedContours;
    }

    /// <summary>
    /// Inflate contour radially from centroid with exponential scaling based on distance.
    /// </summary>
    public static List<Point> InflateContour(List<Point> contour, double inflationAmount, double farPointFactor)
    {
        if (contour.Count == 0) return contour;

        var centroid = new Point2f(
            (float)contour.Average(p => p.X),
            (float)contour.Average(p => p.Y)
        );

        var inflated = new List<Point>();
        var distances = contour.Select(p => Distance(p, centroid)).ToArray();
        var maxDistance = distances.Max();
        if (maxDistance < 1e-6) return contour;

        for (int i = 0; i < contour.Count; i++)
        {
            var p = contour[i];
            var dist = distances[i];
            var normDist = dist / maxDistance;

            // Exponential inflation: inflationAmount * exp((farPointFactor - 1) * normDist)
            var factor = inflationAmount * Math.Exp((farPointFactor - 1) * normDist);

            var direction = new Point2f(p.X - centroid.X, p.Y - centroid.Y);
            var dirNorm = Math.Sqrt(direction.X * direction.X + direction.Y * direction.Y);
            if (dirNorm < 1e-6)
            {
                inflated.Add(p);
                continue;
            }
            direction = new Point2f(direction.X / (float)dirNorm, direction.Y / (float)dirNorm);

            inflated.Add(new Point(
                (int)(p.X + direction.X * factor),
                (int)(p.Y + direction.Y * factor)
            ));
        }

        return inflated;
    }

    /// <summary>
    /// Laplacian smoothing: iteratively average each point with its neighbors.
    /// </summary>
    public static List<Point> LaplacianSmooth(List<Point> contour, int iterations, double alpha)
    {
        if (contour.Count < 3) return contour;

        var smoothed = contour.Select(p => new Point2f(p.X, p.Y)).ToList();
        int n = smoothed.Count;

        for (int iter = 0; iter < iterations; iter++)
        {
            var newSmoothed = new List<Point2f>(smoothed);
            for (int i = 0; i < n; i++)
            {
                var prev = smoothed[(i - 1 + n) % n];
                var next = smoothed[(i + 1) % n];
                var avg = new Point2f((prev.X + next.X) / 2, (prev.Y + next.Y) / 2);
                newSmoothed[i] = new Point2f(
                    (float)((1 - alpha) * smoothed[i].X + alpha * avg.X),
                    (float)((1 - alpha) * smoothed[i].Y + alpha * avg.Y)
                );
            }
            smoothed = newSmoothed;
        }

        return smoothed.Select(p => new Point((int)p.X, (int)p.Y)).ToList();
    }

    /// <summary>
    /// Simplify contour using Douglas-Peucker algorithm.
    /// </summary>
    public static List<Point> SimplifyContour(List<Point> contour, double tolerance)
    {
        if (tolerance <= 0 || contour.Count < 3) return contour;
        var simplified = Cv2.ApproxPolyDP(contour, tolerance, closed: true);
        return simplified.ToList();
    }

    // Helper: RGB to Lab color space (simplified, D65 illuminant)
    private static (double L, double a, double b) RgbToLab(byte r, byte g, byte b)
    {
        // Convert RGB to XYZ
        var rLinear = PivotRgb(r / 255.0);
        var gLinear = PivotRgb(g / 255.0);
        var bLinear = PivotRgb(b / 255.0);

        var x = rLinear * 0.4124564 + gLinear * 0.3575761 + bLinear * 0.1804375;
        var y = rLinear * 0.2126729 + gLinear * 0.7151522 + bLinear * 0.0721750;
        var z = rLinear * 0.0193339 + gLinear * 0.1191920 + bLinear * 0.9503041;

        // Normalize by D65 white point
        x /= 0.95047;
        y /= 1.00000;
        z /= 1.08883;

        // XYZ to Lab
        x = PivotXyz(x);
        y = PivotXyz(y);
        z = PivotXyz(z);

        var L = 116 * y - 16;
        var a = 500 * (x - y);
        var bVal = 200 * (y - z);

        return (L, a, bVal);
    }

    private static double PivotRgb(double n) => n > 0.04045 ? Math.Pow((n + 0.055) / 1.055, 2.4) : n / 12.92;
    private static double PivotXyz(double n) => n > 0.008856 ? Math.Pow(n, 1.0 / 3.0) : (7.787 * n) + (16.0 / 116.0);

    private static double LabDistance((double L, double a, double b) c1, (double L, double a, double b) c2)
    {
        var dL = c1.L - c2.L;
        var da = c1.a - c2.a;
        var db = c1.b - c2.b;
        return Math.Sqrt(dL * dL + da * da + db * db);
    }

    private static double Distance(Point p, Point2f c) =>
        Math.Sqrt(Math.Pow(p.X - c.X, 2) + Math.Pow(p.Y - c.Y, 2));

    private static Point2f Subtract(Point a, Point b) =>
        new Point2f(a.X - b.X, a.Y - b.Y);
}
