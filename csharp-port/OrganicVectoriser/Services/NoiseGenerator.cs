using System;
using OpenCvSharp;

namespace OrganicVectoriser.Services;

/// <summary>
/// Perlin-like noise generator using OpenCV's RNG for simplicity. 
/// For true Perlin, consider LibNoise.NET or porting the Python noise package's pnoise2.
/// This is a simplified gradient-based noise approximation.
/// </summary>
public static class NoiseGenerator
{
    private static readonly Random Rnd = new();

    public static Mat GeneratePerlinNoise(int width, int height, double scale)
    {
        // Simplified gradient noise; real Perlin would interpolate gradients on a grid
        var noise = new Mat(height, width, MatType.CV_32FC1);
        var gridW = Math.Max(1, (int)(width / scale));
        var gridH = Math.Max(1, (int)(height / scale));
        
        // Generate random gradients at grid points
        var gradients = new float[gridH + 1, gridW + 1, 2];
        for (int gy = 0; gy <= gridH; gy++)
        {
            for (int gx = 0; gx <= gridW; gx++)
            {
                var angle = Rnd.NextDouble() * 2 * Math.PI;
                gradients[gy, gx, 0] = (float)Math.Cos(angle);
                gradients[gy, gx, 1] = (float)Math.Sin(angle);
            }
        }

        unsafe
        {
            var ptr = (float*)noise.DataPointer;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var fx = (x / scale);
                    var fy = (y / scale);
                    var gx0 = (int)fx;
                    var gy0 = (int)fy;
                    var gx1 = Math.Min(gx0 + 1, gridW);
                    var gy1 = Math.Min(gy0 + 1, gridH);
                    var sx = fx - gx0;
                    var sy = fy - gy0;

                    // Dot products at four corners
                    var dx0 = sx;
                    var dy0 = sy;
                    var dx1 = sx - 1;
                    var dy1 = sy - 1;
                    var n00 = gradients[gy0, gx0, 0] * dx0 + gradients[gy0, gx0, 1] * dy0;
                    var n10 = gradients[gy0, gx1, 0] * dx1 + gradients[gy0, gx1, 1] * dy0;
                    var n01 = gradients[gy1, gx0, 0] * dx0 + gradients[gy1, gx0, 1] * dy1;
                    var n11 = gradients[gy1, gx1, 0] * dx1 + gradients[gy1, gx1, 1] * dy1;

                    // Smoothstep interpolation
                    var u = Fade(sx);
                    var v = Fade(sy);
                    var nx0 = Lerp(n00, n10, u);
                    var nx1 = Lerp(n01, n11, u);
                    var value = Lerp(nx0, nx1, v);

                    ptr[y * width + x] = (float)value;
                }
            }
        }
        return noise;
    }

    private static double Fade(double t) => t * t * t * (t * (t * 6 - 15) + 10);
    private static double Lerp(double a, double b, double t) => a + t * (b - a);
}
