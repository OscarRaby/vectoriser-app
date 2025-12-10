using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;
using OrganicVectoriser.Models;

namespace OrganicVectoriser.Services;

public static class DropletService
{
    private static readonly Random random = new Random();

    /// <summary>
    /// Generate polygonal approximations of painterly droplets.
    /// Returns droplets as polygon point lists in (row, col) ordering.
    /// </summary>
    public static List<DropletInstance> GenerateDroplets(
        Point2f[] contour,
        Point2d direction,
        int numDroplets,
        double minDist,
        double maxDist,
        double sizeMean,
        double sizeStd,
        double spreadAngle,
        double simplifyTol)
    {
        var droplets = new List<DropletInstance>();
        if (contour == null || contour.Length == 0 || numDroplets <= 0)
            return droplets;

        var meanPt = new Point2d(
            contour.Average(p => p.X),
            contour.Average(p => p.Y)
        );

        double baseAngle = Math.Atan2(direction.Y, direction.X);

        for (int i = 0; i < numDroplets; i++)
        {
            double dist = random.NextDouble() * (maxDist - minDist) + minDist;
            double ang = baseAngle + (random.NextDouble() - 0.5) * spreadAngle;
            double cy = meanPt.Y + Math.Sin(ang) * dist;
            double cx = meanPt.X + Math.Cos(ang) * dist;

            double rx = Math.Abs(RandomNormal(sizeMean, sizeStd));
            double ry = Math.Abs(RandomNormal(sizeMean * 0.8, Math.Max(sizeStd * 0.8, 0.1)));
            rx = Math.Max(0.5, rx);
            ry = Math.Max(0.5, ry);

            // Polygon approximation of an ellipse
            const int nPts = 14;
            var poly = new Point2f[nPts];
            for (int j = 0; j < nPts; j++)
            {
                double angle = 2.0 * Math.PI * j / nPts;
                double y = cy + (Math.Sin(angle) * ry * Math.Cos(ang) - Math.Cos(angle) * rx * Math.Sin(ang));
                double x = cx + (Math.Cos(angle) * rx * Math.Cos(ang) + Math.Sin(angle) * ry * Math.Sin(ang));
                poly[j] = new Point2f((float)x, (float)y);
            }

            // Simplify if requested
            if (simplifyTol > 0)
            {
                var simplified = Cv2.ApproxPolyDP(poly, simplifyTol, true);
                if (simplified.Length > 0)
                    poly = simplified;
            }

            var droplet = new DropletInstance
            {
                Kind = DropletKind.Polygon
            };
            foreach (var pt in poly)
            {
                droplet.Polygon.Add((pt.X, pt.Y));
            }
            droplets.Add(droplet);
        }

        return droplets;
    }

    /// <summary>
    /// Generate lightweight ellipse descriptors for native SVG <ellipse> elements.
    /// Coordinates are in SVG space (cx = column, cy = row).
    /// </summary>
    public static List<DropletInstance> GeneratePainterlyEllipses(
        Point2f[] contour,
        Point2d direction,
        int numDroplets,
        double minDist,
        double maxDist,
        double sizeMean,
        double sizeStd,
        double spreadAngle,
        double rotationOffset)
    {
        var droplets = new List<DropletInstance>();
        if (contour == null || contour.Length == 0 || numDroplets <= 0)
            return droplets;

        var meanPt = new Point2d(
            contour.Average(p => p.X),
            contour.Average(p => p.Y)
        );

        double baseAngle = Math.Atan2(direction.Y, direction.X);

        for (int i = 0; i < numDroplets; i++)
        {
            double dist = random.NextDouble() * (maxDist - minDist) + minDist;
            double ang = baseAngle + (random.NextDouble() - 0.5) * spreadAngle;
            double cy = meanPt.Y + Math.Sin(ang) * dist;
            double cx = meanPt.X + Math.Cos(ang) * dist;

            double rx = Math.Abs(RandomNormal(sizeMean, sizeStd));
            double ry = Math.Abs(RandomNormal(sizeMean * 0.8, Math.Max(sizeStd * 0.8, 0.1)));
            rx = Math.Max(0.5, rx);
            ry = Math.Max(0.5, ry);

            double angleDeg = ang * 180.0 / Math.PI + rotationOffset;

            droplets.Add(new DropletInstance
            {
                Kind = DropletKind.Ellipse,
                Cx = cx,
                Cy = cy,
                Rx = rx,
                Ry = ry,
                AngleDegrees = angleDeg
            });
        }

        return droplets;
    }

    /// <summary>
    /// Generate lightweight rectangle descriptors for native SVG <rect> elements.
    /// Coordinates are in SVG space (cx = column, cy = row).
    /// </summary>
    public static List<DropletInstance> GeneratePainterlyRects(
        Point2f[] contour,
        Point2d direction,
        int numDroplets,
        double minDist,
        double maxDist,
        double sizeMean,
        double sizeStd,
        double spreadAngle,
        double rotationOffset,
        bool horizontal)
    {
        var droplets = new List<DropletInstance>();
        if (contour == null || contour.Length == 0 || numDroplets <= 0)
            return droplets;

        var meanPt = new Point2d(
            contour.Average(p => p.X),
            contour.Average(p => p.Y)
        );

        double baseAngle = Math.Atan2(direction.Y, direction.X);

        for (int i = 0; i < numDroplets; i++)
        {
            double dist = random.NextDouble() * (maxDist - minDist) + minDist;
            double ang = baseAngle + (random.NextDouble() - 0.5) * spreadAngle;
            double cy = meanPt.Y + Math.Sin(ang) * dist;
            double cx = meanPt.X + Math.Cos(ang) * dist;

            double wRect = Math.Abs(RandomNormal(sizeMean * 2.0, sizeStd));
            double hRect = Math.Abs(RandomNormal(sizeMean * 0.8, Math.Max(sizeStd * 0.5, 0.1)));
            wRect = Math.Max(0.5, wRect);
            hRect = Math.Max(0.5, hRect);

            double angleDeg = ang * 180.0 / Math.PI + rotationOffset;

            // If horizontal flag is set, swap width/height so rectangles are elongated horizontally
            double wUse = horizontal ? Math.Max(wRect, hRect) : wRect;
            double hUse = horizontal ? Math.Min(wRect, hRect) : hRect;

            droplets.Add(new DropletInstance
            {
                Kind = DropletKind.Rect,
                Cx = cx,
                Cy = cy,
                Rx = wUse,  // Use Rx to store width
                Ry = hUse,  // Use Ry to store height
                AngleDegrees = angleDeg
            });
        }

        return droplets;
    }

    /// <summary>
    /// Generate a normally distributed random number using Box-Muller transform.
    /// </summary>
    private static double RandomNormal(double mean, double stdDev)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }

    /// <summary>
    /// Generate organic, irregular droplet polygons using Perlin-like noise modulation.
    /// Returns droplets as polygon point lists.
    /// </summary>
    public static List<DropletInstance> GenerateOrganicDroplets(
        Point2f[] contour,
        Point2d direction,
        int numDroplets,
        double minDist,
        double maxDist,
        double sizeMean,
        double sizeStd,
        double spreadAngle,
        double organicStrength,
        double jitterAmount,
        double elongation,
        double simplifyTol)
    {
        var droplets = new List<DropletInstance>();
        if (contour == null || contour.Length == 0 || numDroplets <= 0)
            return droplets;

        // Normalize direction
        var dirLen = Math.Sqrt(direction.X * direction.X + direction.Y * direction.Y);
        if (dirLen < 1e-6)
            direction = new Point2d(1.0, 0.0);
        else
            direction = new Point2d(direction.X / dirLen, direction.Y / dirLen);

        const int nPts = 12;
        var theta = new double[nPts];
        for (int i = 0; i < nPts; i++)
            theta[i] = 2.0 * Math.PI * i / nPts;

        for (int d = 0; d < numDroplets; d++)
        {
            // Random start point on contour
            var startPoint = contour[random.Next(contour.Length)];

            // Random distance and angle jitter
            double dist = random.NextDouble() * (maxDist - minDist) + minDist;
            double angleJitter = (random.NextDouble() - 0.5) * 2.0 * spreadAngle;

            // Rotate direction by jitter
            double c = Math.Cos(angleJitter);
            double s = Math.Sin(angleJitter);
            var dirRot = new Point2d(
                c * direction.X - s * direction.Y,
                s * direction.X + c * direction.Y
            );
            var dirRotLen = Math.Sqrt(dirRot.X * dirRot.X + dirRot.Y * dirRot.Y);
            if (dirRotLen < 1e-6)
                dirRot = direction;
            else
                dirRot = new Point2d(dirRot.X / dirRotLen, dirRot.Y / dirRotLen);

            // Orthogonal direction
            var ortho = new Point2d(-dirRot.Y, dirRot.X);
            var orthoLen = Math.Sqrt(ortho.X * ortho.X + ortho.Y * ortho.Y);
            if (orthoLen < 1e-6)
                ortho = new Point2d(0.0, 1.0);
            else
                ortho = new Point2d(ortho.X / orthoLen, ortho.Y / orthoLen);

            // Center point
            var center = new Point2d(
                startPoint.X + dist * dirRot.X,
                startPoint.Y + dist * dirRot.Y
            );

            // Size with elongation
            double sizeX = Math.Abs(RandomNormal(sizeMean, sizeStd));
            double sizeY = Math.Abs(RandomNormal(sizeMean, sizeStd));

            double sizeMajor, sizeMinor;
            if (elongation > 0)
            {
                sizeMajor = sizeX * (1.0 + elongation);
                sizeMinor = sizeY / (1.0 + elongation);
            }
            else if (elongation < 0)
            {
                double factor = Math.Max(0.01, 1.0 + elongation);
                sizeMajor = sizeX * factor;
                sizeMinor = sizeY / factor;
            }
            else
            {
                sizeMajor = sizeX;
                sizeMinor = sizeY;
            }

            // Generate organic-shaped polygon
            var points = new Point2f[nPts];
            double noiseOffset = random.NextDouble() * 10.0;

            for (int i = 0; i < nPts; i++)
            {
                // Perlin-like noise modulation
                double noiseVal = PerlinNoise1D(theta[i] * 2.0 + noiseOffset);
                double radius = sizeMajor + noiseVal * organicStrength;

                // Elliptical shape with major/minor axes
                double x = center.X + radius * Math.Cos(theta[i]) * dirRot.X + sizeMinor * Math.Sin(theta[i]) * ortho.X;
                double y = center.Y + radius * Math.Cos(theta[i]) * dirRot.Y + sizeMinor * Math.Sin(theta[i]) * ortho.Y;

                // Add jitter
                x += RandomNormal(0, jitterAmount);
                y += RandomNormal(0, jitterAmount);

                points[i] = new Point2f((float)x, (float)y);
            }

            // Simplify if requested
            if (simplifyTol > 0)
            {
                var simplified = Cv2.ApproxPolyDP(points, simplifyTol, true);
                if (simplified.Length >= 3)
                    points = simplified;
            }

            var droplet = new DropletInstance
            {
                Kind = DropletKind.Polygon
            };
            foreach (var pt in points)
            {
                droplet.Polygon.Add((pt.X, pt.Y));
            }
            droplets.Add(droplet);
        }

        return droplets;
    }

    /// <summary>
    /// Simple 1D Perlin-like noise using smoothed interpolation.
    /// </summary>
    private static double PerlinNoise1D(double x)
    {
        int x0 = (int)Math.Floor(x);
        int x1 = x0 + 1;
        double t = x - x0;

        // Smoothstep interpolation
        t = t * t * (3.0 - 2.0 * t);

        // Pseudo-random gradients based on integer coordinates
        double g0 = GradientNoise(x0);
        double g1 = GradientNoise(x1);

        return g0 * (1.0 - t) + g1 * t;
    }

    /// <summary>
    /// Generate pseudo-random gradient value for a given integer coordinate.
    /// </summary>
    private static double GradientNoise(int x)
    {
        // Simple hash-based pseudo-random number
        x = (x << 13) ^ x;
        int n = (x * (x * x * 15731 + 789221) + 1376312589) & 0x7fffffff;
        return 1.0 - (n / 1073741824.0);
    }
}
