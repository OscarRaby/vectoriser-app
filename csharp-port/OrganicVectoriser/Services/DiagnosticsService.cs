using System;
using System.Collections.Generic;
using System.IO;
using OpenCvSharp;
using OrganicVectoriser.Models;

namespace OrganicVectoriser.Services;

/// <summary>
/// Diagnostic service for isolating and testing individual pipeline stages
/// </summary>
public interface IDiagnosticsService
{
    /// <summary>Test noise generation in isolation</summary>
    Mat TestNoiseGeneration(int width, int height, double scale);

    /// <summary>Test elevation map creation</summary>
    Mat TestElevationMap(Mat rgb, double blurSigma, double noiseScale);

    /// <summary>Test local minima detection</summary>
    Mat TestLocalMinima(Mat elevationMap);

    /// <summary>Test watershed segmentation</summary>
    Mat TestWatershed(Mat rgb, Mat elevationMap);

    /// <summary>Save a Mat to disk for visual inspection</summary>
    void SaveMatAsImage(Mat mat, string filename, string stage);

    /// <summary>Get statistics about labels in a Mat</summary>
    string GetLabelStatistics(Mat labels);
}

public sealed class DiagnosticsService : IDiagnosticsService
{
    private static readonly string DiagFolder = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "OrganicVectoriser",
        "diagnostics"
    );

    private static readonly string DiagLogFile = Path.Combine(DiagFolder, "diagnostics.log");

    public DiagnosticsService()
    {
        Directory.CreateDirectory(DiagFolder);
    }

    private void Log(string message)
    {
        var line = $"[{DateTime.Now:HH:mm:ss.fff}] {message}";
        Console.WriteLine(line);
        try
        {
            File.AppendAllText(DiagLogFile, line + "\n");
        }
        catch { /* Ignore write errors */ }
    }

    public Mat TestNoiseGeneration(int width, int height, double scale)
    {
        Log($"[DIAG] Testing noise generation: {width}x{height}, scale={scale}");
        
        var noise = NoiseGenerator.GeneratePerlinNoise(width, height, scale);
        
        // Get statistics
        Cv2.MinMaxLoc(noise, out double minVal, out double maxVal);
        Log($"[DIAG] Noise value range: [{minVal:F4}, {maxVal:F4}]");
        
        // Normalize to 0-255 for visualization
        var noiseVisualized = new Mat();
        Cv2.Normalize(noise, noiseVisualized, 0, 255, NormTypes.MinMax);
        noiseVisualized.ConvertTo(noiseVisualized, MatType.CV_8UC1);
        
        SaveMatAsImage(noiseVisualized, $"01_noise_scale{scale:F0}.png", "NOISE");
        return noise;
    }

    public Mat TestElevationMap(Mat rgb, double blurSigma, double noiseScale)
    {
        Log($"[DIAG] Testing elevation map: blur_sigma={blurSigma}, noise_scale={noiseScale}");
        
        // Convert to grayscale
        using var gray = new Mat();
        Cv2.CvtColor(rgb, gray, ColorConversionCodes.RGB2GRAY);
        gray.ConvertTo(gray, MatType.CV_32FC1, 1.0 / 255.0);
        Log($"[DIAG] Gray: min={gray.Get<float>(0, 0):F4}, max={gray.Get<float>(gray.Rows - 1, gray.Cols - 1):F4}");

        // Gaussian blur
        using var blurred = new Mat();
        var ksize = (int)(blurSigma * 6) | 1;
        Cv2.GaussianBlur(gray, blurred, new OpenCvSharp.Size(ksize, ksize), blurSigma);
        Log($"[DIAG] Blurred with ksize={ksize}");

        // Generate noise
        using var noise = NoiseGenerator.GeneratePerlinNoise(rgb.Width, rgb.Height, noiseScale);
        
        // Combine: elevation = 0.7 * blurred + 0.3 * noise
        var elevation = new Mat();
        Cv2.AddWeighted(blurred, 0.7, noise, 0.3, 0, elevation);
        
        Cv2.MinMaxLoc(elevation, out double minVal, out double maxVal);
        var mean = Cv2.Mean(elevation);
        
        using var meanMat = new Mat();
        using var stdDevMat = new Mat();
        Cv2.MeanStdDev(elevation, meanMat, stdDevMat);
        double stdDev = stdDevMat.At<double>(0, 0);
        
        Log($"[DIAG] Elevation range: [{minVal:F4}, {maxVal:F4}]");
        Log($"[DIAG] Elevation mean: {mean.Val0:F4}, std dev: {stdDev:F4}");
        Log($"[DIAG] Elevation variation: {(maxVal - minVal):F4}");
        
        // Check if elevation has sufficient variation
        if ((maxVal - minVal) < 0.1)
        {
            Log($"[DIAG] WARNING: Elevation map has very little variation (<0.1)!");
            Log($"[DIAG] This will cause poor watershed segmentation.");
        }
        
        // Visualize
        var elevationVisualized = new Mat();
        Cv2.Normalize(elevation, elevationVisualized, 0, 255, NormTypes.MinMax);
        elevationVisualized.ConvertTo(elevationVisualized, MatType.CV_8UC1);
        SaveMatAsImage(elevationVisualized, $"02_elevation_blur{blurSigma:F1}_noise{noiseScale:F0}.png", "ELEVATION");
        
        return elevation;
    }

    public Mat TestLocalMinima(Mat elevationMap)
    {
        Log($"[DIAG] Testing local minima detection on elevation map");
        
        // Convert to 8U
        using var elevationU8 = new Mat();
        elevationMap.ConvertTo(elevationU8, MatType.CV_8UC1, 255);
        
        // Create a binary mask by thresholding at mean - this creates foreground regions
        var mean = Cv2.Mean(elevationU8);
        using var binary = new Mat();
        Cv2.Threshold(elevationU8, binary, mean.Val0, 255, ThresholdTypes.Binary);
        Log($"[DIAG] Created binary mask with threshold at mean ({mean.Val0:F2})");
        
        // Invert so we have white foreground (high elevation areas)
        Cv2.BitwiseNot(binary, binary);
        
        // Distance transform on binary image
        using var dist = new Mat();
        Cv2.DistanceTransform(binary, dist, DistanceTypes.L2, DistanceTransformMasks.Mask5);
        
        Cv2.MinMaxLoc(dist, out double minDist, out double maxDist);
        Log($"[DIAG] Distance transform range: [{minDist:F2}, {maxDist:F2}]");
        
        if (maxDist > 1e10)
        {
            Log($"[DIAG] ERROR: Distance transform returned invalid values (infinity)!");
            Log($"[DIAG] This indicates the binary mask is all white or all black");
            int whitePixels = Cv2.CountNonZero(binary);
            Log($"[DIAG] Binary mask has {whitePixels} white pixels ({(whitePixels * 100.0 / (elevationMap.Rows * elevationMap.Cols)):F2}%)");
        }
        
        // Save distance transform for visualization
        var distViz = new Mat();
        Cv2.Normalize(dist, distViz, 0, 255, NormTypes.MinMax);
        distViz.ConvertTo(distViz, MatType.CV_8UC1);
        SaveMatAsImage(distViz, "03a_distance_transform.png", "DISTANCE");
        SaveMatAsImage(binary, "03a_binary_mask.png", "BINARY");
        distViz.Dispose();
        
        // Threshold to find peaks - try multiple thresholds to see what works
        Log($"[DIAG] Testing different thresholds:");
        
        for (double factor = 0.7; factor >= 0.1; factor -= 0.2)
        {
            using var testMax = new Mat();
            double threshold = factor * maxDist;
            Cv2.Threshold(dist, testMax, threshold, 255, ThresholdTypes.Binary);
            testMax.ConvertTo(testMax, MatType.CV_8UC1);
            int count = Cv2.CountNonZero(testMax);
            Log($"[DIAG]   Threshold {factor:F1} * maxDist ({threshold:F2}): {count} pixels ({(count * 100.0 / (elevationMap.Rows * elevationMap.Cols)):F2}%)");
        }
        
        // Use 0.5 * maxDist as threshold (more selective than 0.3)
        using var localMax = new Mat();
        double finalThreshold = 0.5 * maxDist;
        Cv2.Threshold(dist, localMax, finalThreshold, 255, ThresholdTypes.Binary);
        localMax.ConvertTo(localMax, MatType.CV_8UC1);
        
        Log($"[DIAG] Final threshold used: {finalThreshold:F2}");
        
        // Count non-zero pixels (potential marker points)
        int markerCount = Cv2.CountNonZero(localMax);
        Log($"[DIAG] Found {markerCount} potential marker points ({(markerCount * 100.0 / (elevationMap.Rows * elevationMap.Cols)):F2}% of image)");
        
        SaveMatAsImage(localMax, "03b_local_maxima.png", "MARKERS");
        return localMax;
    }

    public Mat TestWatershed(Mat rgb, Mat elevationMap)
    {
        Log($"[DIAG] Testing watershed segmentation");
        
        // Use the same binary mask approach as TestLocalMinima
        using var elevationU8 = new Mat();
        elevationMap.ConvertTo(elevationU8, MatType.CV_8UC1, 255);
        
        var mean = Cv2.Mean(elevationU8);
        using var binary = new Mat();
        Cv2.Threshold(elevationU8, binary, mean.Val0, 255, ThresholdTypes.Binary);
        Cv2.BitwiseNot(binary, binary);
        
        using var dist = new Mat();
        Cv2.DistanceTransform(binary, dist, DistanceTypes.L2, DistanceTransformMasks.Mask5);
        Cv2.MinMaxLoc(dist, out _, out double maxVal);
        
        Log($"[DIAG] Watershed distance transform max: {maxVal:F2}");
        
        using var localMax = new Mat();
        Cv2.Threshold(dist, localMax, 0.5 * maxVal, 255, ThresholdTypes.Binary);
        localMax.ConvertTo(localMax, MatType.CV_8UC1);
        
        // Get connected components
        var markers = new Mat();
        int numComponents = Cv2.ConnectedComponents(localMax, markers);
        markers.ConvertTo(markers, MatType.CV_32SC1);
        Log($"[DIAG] Connected components found: {numComponents}");
        
        // Apply watershed
        Log($"[DIAG] Applying watershed algorithm...");
        var rgbCopy = rgb.Clone();
        Cv2.Watershed(rgbCopy, markers);
        rgbCopy.Dispose();
        
        // Count unique labels in result
        var uniqueLabels = new HashSet<int>();
        for (int y = 0; y < markers.Rows; y++)
        {
            for (int x = 0; x < markers.Cols; x++)
            {
                int label = markers.Get<int>(y, x);
                uniqueLabels.Add(label);
            }
        }
        
        Log($"[DIAG] Watershed output unique labels: {uniqueLabels.Count}");
        if (uniqueLabels.Contains(-1))
            Log($"[DIAG] WARNING: Found -1 labels (watershed edges)");
        
        // Visualize - need to convert CV_32SC1 to CV_8UC1 for ApplyColorMap
        var markers8U = new Mat();
        markers.ConvertTo(markers8U, MatType.CV_8UC1);
        var markerViz = new Mat();
        Cv2.ApplyColorMap(markers8U, markerViz, ColormapTypes.Jet);
        SaveMatAsImage(markerViz, "04_watershed_result.png", "WATERSHED");
        markerViz.Dispose();
        markers8U.Dispose();
        
        return markers;
    }

    public void SaveMatAsImage(Mat mat, string filename, string stage)
    {
        try
        {
            string fullPath = Path.Combine(DiagFolder, filename);
            
            // Convert to 8UC3 if needed for saving
            Mat toSave = mat;
            Mat? converted = null;
            
            if (mat.Type() == MatType.CV_32FC1)
            {
                converted = new Mat();
                Cv2.Normalize(mat, converted, 0, 255, NormTypes.MinMax);
                converted.ConvertTo(converted, MatType.CV_8UC1);
                toSave = converted;
            }
            else if (mat.Type() == MatType.CV_32SC1)
            {
                // Colormap for label visualization
                converted = new Mat();
                Cv2.ApplyColorMap(mat, converted, ColormapTypes.Jet);
                toSave = converted;
            }
            
            Cv2.ImWrite(fullPath, toSave);
            Log($"[DIAG] [{stage}] Saved: {fullPath}");
            
            converted?.Dispose();
        }
        catch (Exception ex)
        {
            Log($"[DIAG] Error saving image: {ex.Message}");
        }
    }

    public string GetLabelStatistics(Mat labels)
    {
        var stats = new System.Text.StringBuilder();
        var labelCounts = new Dictionary<int, int>();
        
        for (int y = 0; y < labels.Rows; y++)
        {
            for (int x = 0; x < labels.Cols; x++)
            {
                int label = labels.Get<int>(y, x);
                if (!labelCounts.ContainsKey(label))
                    labelCounts[label] = 0;
                labelCounts[label]++;
            }
        }
        
        stats.AppendLine($"Label Statistics for {labels.Rows}x{labels.Cols} image:");
        stats.AppendLine($"  Total unique labels: {labelCounts.Count}");
        
        foreach (var kvp in labelCounts)
        {
            float percentage = (kvp.Value * 100f) / (labels.Rows * labels.Cols);
            stats.AppendLine($"  Label {kvp.Key}: {kvp.Value} pixels ({percentage:F2}%)");
        }
        
        return stats.ToString();
    }
}

