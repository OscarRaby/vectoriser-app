using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using Accord.MachineLearning;
using OpenCvSharp;
using OrganicVectoriser.Models;

namespace OrganicVectoriser.Services;

public interface IPipelineService
{
    Task<PipelineResult> RunAsync(BitmapInput input, ParameterSet parameters, ModifierFlags modifiers);
    Task<PipelineResult> PreviewAsync(BitmapInput input, ParameterSet parameters, ModifierFlags modifiers);
}

public sealed class PipelineService : IPipelineService
{
    public Task<PipelineResult> RunAsync(BitmapInput input, ParameterSet parameters, ModifierFlags modifiers)
    {
        return Task.Run(() => ExecutePipeline(input, parameters, modifiers, fullRun: true));
    }

    public Task<PipelineResult> PreviewAsync(BitmapInput input, ParameterSet parameters, ModifierFlags modifiers)
    {
        return Task.Run(() => ExecutePipeline(input, parameters, modifiers, fullRun: false));
    }

    private PipelineResult ExecutePipeline(BitmapInput input, ParameterSet p, ModifierFlags m, bool fullRun)
    {
        try
        {
            var opType = fullRun ? "Run" : "Preview";
            LogPipeline($"[{opType}] Starting pipeline execution");
            
            var result = new PipelineResult { Width = input.Width, Height = input.Height };

            if (input.Width <= 0 || input.Height <= 0)
                throw new ArgumentException("Invalid image dimensions");

            LogPipeline($"[{opType}] Image dimensions: {input.Width}x{input.Height}");

            // Convert input to OpenCV Mat (assume BGR24 or RGB24)
            LogPipeline($"[{opType}] Converting pixel data to Mat");
            using var mat = Mat.FromPixelData(input.Height, input.Width, MatType.CV_8UC3, input.Pixels);
            
            // Ensure we have a valid Mat
            if (mat.Empty())
                throw new InvalidOperationException("Failed to create Mat from pixel data");

            LogPipeline($"[{opType}] Converting BGR to RGB");
            var rgb = new Mat();
            Cv2.CvtColor(mat, rgb, ColorConversionCodes.BGR2RGB);

            if (rgb.Empty())
                throw new InvalidOperationException("Failed to convert color space");

            // Scale parameters by image size (inverse scaling for noise to maintain grain size)
            var scaleFactor = GetScaleFactor(input.Width, input.Height);
            var multiplier = Math.Max(p.SegmentationMultiplier, 1e-6);
            // Noise scale should DECREASE for larger images to maintain same grain size
            // This matches Python's pnoise2 behavior where scale is a frequency divisor
            var noiseScale = Math.Max(1.0, p.NoiseScale * multiplier / scaleFactor);
            var blurSigma = Math.Max(1e-6, p.BlurSigma * scaleFactor / multiplier);
            var compactness = Math.Max(1e-6, p.Compactness * scaleFactor / multiplier);

            // 1. Segmentation: noise-modulated watershed
            LogPipeline($"[{opType}] About to call NoiseWatershed with noiseScale={noiseScale}, blurSigma={blurSigma}, compactness={compactness}");
            var labels = NoiseWatershed(rgb, noiseScale, blurSigma, compactness);
            LogPipeline($"[{opType}] NoiseWatershed completed");
            

            // 2. Color quantization (optional)
            LogPipeline($"[{opType}] About to call QuantizeColors, quantization enabled: {m.ColorQuantization}");
            using var quantized = m.ColorQuantization
                ? QuantizeColors(rgb, p.MaxColors)
                : rgb.Clone();
            LogPipeline($"[{opType}] QuantizeColors completed");
            
            // 3. Extract contours and colors
            LogPipeline($"[{opType}] About to call ExtractContours");
            var (contours, centroids, colors) = ExtractContours(labels, quantized);
            LogPipeline($"[{opType}] ExtractContours completed with {contours.Count} contours");
            
            // Dispose Mats after use
            labels.Dispose();
            rgb.Dispose();
            

            // 4. Bridging (optional)
            if (m.Bridging)
            {
                
                contours = ContourService.BridgeContours(
                    contours,
                    centroids,
                    colors,
                    p.BridgeDistance,
                    p.ColorTolerance,
                    p.ProximityThreshold,
                    p.FalloffRadius,
                    Math.PI * p.MaxCurvatureDegrees / 180.0
                );
                
            }

            // 5. Inflation (optional)
            if (m.Inflation)
            {
                
                var N = contours.Count;
                for (int idx = 0; idx < N; idx++)
                {
                    var inflationAmount = p.BlobInflationAmount;
                    var farPointFactor = p.FarPointInflationFactor;

                    // Apply stacking-proportional scaling if enabled
                    if (p.InflationProportionalToStacking && N > 1)
                    {
                        var stackScale = Math.Log(1 + idx) / Math.Log(N);
                        inflationAmount *= stackScale;
                        farPointFactor = 1.0 + (farPointFactor - 1.0) * stackScale;
                    }

                    contours[idx] = ContourService.InflateContour(contours[idx], inflationAmount, farPointFactor);
                }
                
            }

            // 6. Smoothing (optional)
            if (m.Smoothing)
            {
                
                for (int idx = 0; idx < contours.Count; idx++)
                {
                    contours[idx] = ContourService.LaplacianSmooth(contours[idx], p.SmoothIterations, p.SmoothAlpha);
                }
                
            }

            // 7. Simplification (always applied if tolerance > 0)
            if (p.SimplifyTolerance > 0)
            {
                
                for (int idx = 0; idx < contours.Count; idx++)
                {
                    contours[idx] = ContourService.SimplifyContour(contours[idx], p.SimplifyTolerance);
                }
                
            }

            // 7. Droplets (skip for preview)
            if (fullRun && (p.DropletDensity > 0 || (p.DropletStyle == "organic" && p.DropletOrganicDensity > 0)))
            {
                
                var spreadAngle = p.DropletSpreadDegrees * Math.PI / 180.0;

                if (p.DropletStyle == "organic")
                {
                    
                    // Organic droplets: generate between neighboring blobs based on brightness
                    for (int i = 0; i < contours.Count; i++)
                    {
                        if (contours[i].Count == 0)
                            continue;

                        var color = colors[i];
                        var luminance = Luminance(color);

                        // Skip if brightness is below threshold
                        if (luminance < p.DropletOrganicMinBrightness)
                            continue;

                    var currentCentroid = centroids[i];
                    var contour = contours[i].Select(pt => new OpenCvSharp.Point2f(pt.X, pt.Y)).ToArray();

                    // Find neighbor indices (all except self)
                    var neighborIndices = new List<int>();
                    for (int j = 0; j < contours.Count; j++)
                    {
                        if (j != i)
                            neighborIndices.Add(j);
                    }

                    if (neighborIndices.Count == 0)
                        continue;

                    // Select a percentage of neighbors
                    int nSelected = Math.Max(1, (int)Math.Ceiling((p.DropletOrganicPercentPerBlob / 100.0) * neighborIndices.Count));
                    nSelected = Math.Min(nSelected, neighborIndices.Count);

                    // Random selection of neighbors
                    var selectedNeighbors = neighborIndices.OrderBy(x => Guid.NewGuid()).Take(nSelected).ToList();

                    foreach (var j in selectedNeighbors)
                    {
                        var neighborCentroid = centroids[j];
                        var direction = new OpenCvSharp.Point2d(
                            neighborCentroid.X - currentCentroid.X,
                            neighborCentroid.Y - currentCentroid.Y
                        );
                        var norm = Math.Sqrt(direction.X * direction.X + direction.Y * direction.Y);
                        if (norm < 1e-6)
                            continue;
                        direction = new OpenCvSharp.Point2d(direction.X / norm, direction.Y / norm);

                        var droplets = DropletService.GenerateOrganicDroplets(
                            contour,
                            direction,
                            p.DropletOrganicDensity,
                            p.DropletMinDistance,
                            p.DropletMaxDistance,
                            p.DropletSizeMean,
                            p.DropletSizeStd,
                            spreadAngle,
                            p.DropletOrganicStrength,
                            p.DropletOrganicJitter,
                            p.DropletOrganicElongation,
                            p.SimplifyTolerance
                        );

                        foreach (var droplet in droplets)
                        {
                            droplet.FillColor = color;
                            result.Droplets.Add(droplet);
                        }
                    }
                }
            }
            }
            else
            {
                // Painterly droplets: generate from each contour
                var spreadAngle = p.DropletSpreadDegrees * Math.PI / 180.0;
                for (int idx = 0; idx < contours.Count; idx++)
                {
                    var contour = contours[idx].Select(pt => new OpenCvSharp.Point2f(pt.X, pt.Y)).ToArray();
                    var centroid = centroids[idx];
                    var color = colors[idx];

                    // Calculate direction vector from centroid to mean point of contour
                    var meanPt = new OpenCvSharp.Point2d(
                        contour.Average(pt => pt.X),
                        contour.Average(pt => pt.Y)
                    );
                    var direction = new OpenCvSharp.Point2d(meanPt.X - centroid.X, meanPt.Y - centroid.Y);
                    var norm = Math.Sqrt(direction.X * direction.X + direction.Y * direction.Y);
                    if (norm < 1e-6)
                        direction = new OpenCvSharp.Point2d(1.0, 0.0);
                    else
                        direction = new OpenCvSharp.Point2d(direction.X / norm, direction.Y / norm);

                    List<DropletInstance> droplets;

                    // Check droplet style from modifier flags or parameters
                    // For now, we'll check PainterlyUseSvgEllipses to determine which method to use
                    if (p.PainterlyUseSvgEllipses)
                    {
                        // Native SVG primitives (ellipse or rect)
                        if (p.PainterlySvgPrimitive == "ellipse")
                        {
                            droplets = DropletService.GeneratePainterlyEllipses(
                                contour,
                                direction,
                                p.DropletDensity,
                                p.DropletMinDistance,
                                p.DropletMaxDistance,
                                p.DropletSizeMean,
                                p.DropletSizeStd,
                                spreadAngle,
                                p.DropletGlobalRotation
                            );
                        }
                        else // rect
                        {
                            droplets = DropletService.GeneratePainterlyRects(
                                contour,
                                direction,
                                p.DropletDensity,
                                p.DropletMinDistance,
                                p.DropletMaxDistance,
                                p.DropletSizeMean,
                                p.DropletSizeStd,
                                spreadAngle,
                                p.DropletGlobalRotation,
                                p.PainterlyRectHorizontal
                            );
                        }
                    }
                    else
                    {
                        // Polygonal approximation
                        droplets = DropletService.GenerateDroplets(
                            contour,
                            direction,
                            p.DropletDensity,
                            p.DropletMinDistance,
                            p.DropletMaxDistance,
                            p.DropletSizeMean,
                            p.DropletSizeStd,
                            spreadAngle,
                            p.SimplifyTolerance
                        );
                    }

                    // Add color to each droplet and add to result
                    foreach (var droplet in droplets)
                    {
                        droplet.FillColor = color;
                        result.Droplets.Add(droplet);
                    }
                }
                
            }

            // 8. Stacking order
            
            var indices = ApplyStackingOrder(contours, centroids, colors, p.StackingOrder, input.Width, input.Height);

            // Convert contours to result format (in stacking order)
            
            foreach (var idx in indices)
            {
                var contourData = new ContourData { FillColor = colors[idx] };
                foreach (var pt in contours[idx])
                {
                    contourData.Points.Add((pt.X, pt.Y));
                }
                result.Contours.Add(contourData);
            }

            return result;
        }
        catch (TypeInitializationException tie) when (tie.InnerException?.Message.Contains("NativeMethods") ?? false)
        {
            
            throw new InvalidOperationException(
                $"OpenCV native library initialization failed. This typically means OpenCvSharp cannot load its native dependencies. " +
                $"Error: {tie.InnerException?.Message}",
                tie);
        }
        catch (Exception)
        {
            
            throw;
        }
    }

    private int[] ApplyStackingOrder(
        List<List<OpenCvSharp.Point>> contours,
        List<OpenCvSharp.Point2f> centroids,
        List<(byte, byte, byte)> colors,
        string stackingOrder,
        int width,
        int height)
    {
        var indices = Enumerable.Range(0, contours.Count).ToArray();
        if (contours.Count == 0) return indices;

        switch (stackingOrder)
        {
            case "area":
                var areas = contours.Select(c => ContourArea(c)).ToArray();
                return indices.OrderByDescending(i => areas[i]).ToArray();

            case "area_reverse":
                areas = contours.Select(c => ContourArea(c)).ToArray();
                return indices.OrderBy(i => areas[i]).ToArray();

            case "brightness":
                var luminances = colors.Select(c => Luminance(c)).ToArray();
                return indices.OrderBy(i => luminances[i]).ToArray();

            case "brightness_reverse":
                luminances = colors.Select(c => Luminance(c)).ToArray();
                return indices.OrderByDescending(i => luminances[i]).ToArray();

            case "position_y":
                var ys = centroids.Select(c => c.Y).ToArray();
                return indices.OrderBy(i => ys[i]).ToArray();

            case "position_y_reverse":
                ys = centroids.Select(c => c.Y).ToArray();
                return indices.OrderByDescending(i => ys[i]).ToArray();

            case "position_x":
                var xs = centroids.Select(c => c.X).ToArray();
                return indices.OrderBy(i => xs[i]).ToArray();

            case "position_x_reverse":
                xs = centroids.Select(c => c.X).ToArray();
                return indices.OrderByDescending(i => xs[i]).ToArray();

            case "position_centre":
                var center = new OpenCvSharp.Point2f(width / 2f, height / 2f);
                var dists = centroids.Select(c => Distance(c, center)).ToArray();
                return indices.OrderByDescending(i => dists[i]).ToArray();

            case "position_centre_reverse":
                center = new OpenCvSharp.Point2f(width / 2f, height / 2f);
                dists = centroids.Select(c => Distance(c, center)).ToArray();
                return indices.OrderBy(i => dists[i]).ToArray();

            default:
                return indices;
        }
    }

    private double ContourArea(List<OpenCvSharp.Point> contour)
    {
        if (contour.Count < 3) return 0;
        var xs = contour.Select(p => (double)p.X).ToArray();
        var ys = contour.Select(p => (double)p.Y).ToArray();
        var sum = 0.0;
        for (int i = 0; i < contour.Count; i++)
        {
            var j = (i + 1) % contour.Count;
            sum += xs[i] * ys[j] - xs[j] * ys[i];
        }
        return Math.Abs(sum) / 2.0;
    }

    private double Luminance((byte R, byte G, byte B) c) => 0.2126 * c.R + 0.7152 * c.G + 0.0722 * c.B;

    private double Distance(OpenCvSharp.Point2f a, OpenCvSharp.Point2f b)
    {
        var dx = a.X - b.X;
        var dy = a.Y - b.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    private double GetScaleFactor(int width, int height)
    {
        const int refW = 150, refH = 150;
        return Math.Max(width, height) / (double)Math.Max(refW, refH);
    }

    private Mat NoiseWatershed(Mat rgb, double noiseScale, double blurSigma, double compactness)
    {
        try
        {
            LogPipeline($"[NoiseWatershed] Starting with noise={noiseScale}, blur={blurSigma}, compact={compactness}");
            
            // Validate input
            if (rgb.Type() != MatType.CV_8UC3)
                throw new ArgumentException($"Expected CV_8UC3 input, got {rgb.Type()}");

            LogPipeline($"[NoiseWatershed] Input validation passed");

            using var gray = new Mat();
            Cv2.CvtColor(rgb, gray, ColorConversionCodes.RGB2GRAY);
            gray.ConvertTo(gray, MatType.CV_32FC1, 1.0 / 255.0);
            LogPipeline($"[NoiseWatershed] Grayscale conversion done");

            // Gaussian blur
            using var blurred = new Mat();
            var ksize = (int)(blurSigma * 6) | 1;
            LogPipeline($"[NoiseWatershed] Gaussian blur with ksize={ksize}");
            Cv2.GaussianBlur(gray, blurred, new OpenCvSharp.Size(ksize, ksize), blurSigma);
            LogPipeline($"[NoiseWatershed] Gaussian blur done");

            // Generate Perlin noise field
            LogPipeline($"[NoiseWatershed] Generating Perlin noise {rgb.Width}x{rgb.Height}");
            using var noise = NoiseGenerator.GeneratePerlinNoise(rgb.Width, rgb.Height, noiseScale);
            LogPipeline($"[NoiseWatershed] Perlin noise generated");

            // Combine: elevation = 0.7 * blurred + 0.3 * noise
            using var elevation = new Mat();
            Cv2.AddWeighted(blurred, 0.7, noise, 0.3, 0, elevation);
            LogPipeline($"[NoiseWatershed] Elevation map combined");

            // Find local minima (distance transform approach or morphological min filter)
            using var elevationU8 = new Mat();
            elevation.ConvertTo(elevationU8, MatType.CV_8UC1, 255);
            Cv2.BitwiseNot(elevationU8, elevationU8);
            using var dist = new Mat();
            Cv2.DistanceTransform(elevationU8, dist, DistanceTypes.L2, DistanceTransformMasks.Mask5);
            using var localMax = new Mat();
            Cv2.MinMaxLoc(dist, out _, out double maxVal);
            Cv2.Threshold(dist, localMax, 0.3 * maxVal, 255, ThresholdTypes.Binary);
            localMax.ConvertTo(localMax, MatType.CV_8UC1);
            LogPipeline($"[NoiseWatershed] Local maxima found");

            // Label markers
            var markers = new Mat();
            Cv2.ConnectedComponents(localMax, markers);
            markers.ConvertTo(markers, MatType.CV_32SC1);
            LogPipeline($"[NoiseWatershed] Connected components labeled");

            // Watershed - apply to original RGB image with markers
            LogPipeline($"[NoiseWatershed] About to call Watershed");
            var rgbCopy = rgb.Clone();  // watershed modifies markers in place, so we can keep original
            Cv2.Watershed(rgbCopy, markers);
            rgbCopy.Dispose();
            LogPipeline($"[NoiseWatershed] Watershed completed");

            return markers;
        }
        catch (Exception ex)
        {
            LogPipeline($"[NoiseWatershed] EXCEPTION: {ex.GetType().Name}: {ex.Message}");
            throw;
        }
    }

    private Mat QuantizeColors(Mat rgb, int maxColors)
    {
        try
        {
            LogPipeline($"[QuantizeColors] Starting with {rgb.Rows}x{rgb.Cols}, maxColors={maxColors}");
            var pixelCount = rgb.Rows * rgb.Cols;
            LogPipeline($"[QuantizeColors] Extracting {pixelCount} pixels");
            var pixels = new double[pixelCount][];
            int idx = 0;
            for (int y = 0; y < rgb.Rows; y++)
            {
                for (int x = 0; x < rgb.Cols; x++)
                {
                    var color = rgb.At<Vec3b>(y, x);
                    pixels[idx] = new double[] { color.Item0, color.Item1, color.Item2 };
                    idx++;
                }
            }
            LogPipeline($"[QuantizeColors] Pixels extracted, running KMeans");

            var kmeans = new KMeans(maxColors) { MaxIterations = 10 };
            var clusters = kmeans.Learn(pixels);
            LogPipeline($"[QuantizeColors] KMeans learned");
            
            var labels = clusters.Decide(pixels);
            LogPipeline($"[QuantizeColors] Labels decided");
            
            var centroids = kmeans.Centroids;
            LogPipeline($"[QuantizeColors] Creating quantized image");

            var quantized = new Mat(rgb.Size(), MatType.CV_8UC3);
            idx = 0;
            for (int y = 0; y < rgb.Rows; y++)
            {
                for (int x = 0; x < rgb.Cols; x++)
                {
                    var label = labels[idx++];
                    var c = centroids[label];
                    quantized.Set(y, x, new Vec3b((byte)c[0], (byte)c[1], (byte)c[2]));
                }
            }
            LogPipeline($"[QuantizeColors] Quantization complete");
            return quantized;
        }
        catch (Exception ex)
        {
            LogPipeline($"[QuantizeColors] EXCEPTION: {ex.GetType().Name}: {ex.Message}");
            throw;
        }
    }

    private (List<List<OpenCvSharp.Point>> contours, List<OpenCvSharp.Point2f> centroids, List<(byte, byte, byte)> colors) ExtractContours(Mat labels, Mat quantized)
    {
        try
        {
            LogPipeline($"[ExtractContours] Function entered");
            
            // Validate Mats
            if (labels == null || labels.Empty())
            {
                LogPipeline($"[ExtractContours] ERROR: labels Mat is null or empty!");
                throw new ArgumentException("labels Mat is invalid");
            }
            if (quantized == null || quantized.Empty())
            {
                LogPipeline($"[ExtractContours] ERROR: quantized Mat is null or empty!");
                throw new ArgumentException("quantized Mat is invalid");
            }
            
            LogPipeline($"[ExtractContours] Starting with labels={labels.Rows}x{labels.Cols}, quantized={quantized.Rows}x{quantized.Cols}");
            var contours = new List<List<OpenCvSharp.Point>>();
            var centroids = new List<OpenCvSharp.Point2f>();
            var colors = new List<(byte, byte, byte)>();

            LogPipeline($"[ExtractContours] Finding unique labels");
            var uniqueLabels = new HashSet<int>();
            for (int y = 0; y < labels.Rows; y++)
            {
                for (int x = 0; x < labels.Cols; x++)
                {
                    var label = labels.At<int>(y, x);
                    if (label > 0) uniqueLabels.Add(label);
                }
            }
            LogPipeline($"[ExtractContours] Found {uniqueLabels.Count} unique labels");

            int segmentCount = 0;
            foreach (var segId in uniqueLabels)
            {
                segmentCount++;
                if (segmentCount % 10 == 0)
                    LogPipeline($"[ExtractContours] Processing segment {segmentCount}/{uniqueLabels.Count}");
                    
                using var mask = new Mat(labels.Size(), MatType.CV_8UC1, Scalar.Black);
                for (int y = 0; y < labels.Rows; y++)
                {
                    for (int x = 0; x < labels.Cols; x++)
                    {
                        if (labels.At<int>(y, x) == segId)
                        {
                            mask.Set(y, x, (byte)255);
                        }
                    }
                }

                // Find contours
                Cv2.FindContours(mask, out var foundContours, out _, RetrievalModes.External, ContourApproximationModes.ApproxNone);
                if (foundContours.Length == 0) continue;
                var contour = foundContours[0].ToList();
                contours.Add(contour);

                // Compute centroid
                var moments = Cv2.Moments(contour.ToArray());
                var cx = moments.M10 / moments.M00;
                var cy = moments.M01 / moments.M00;
                centroids.Add(new OpenCvSharp.Point2f((float)cx, (float)cy));

                // Majority color
                var colorCounts = new Dictionary<(byte, byte, byte), int>();
                for (int y = 0; y < labels.Rows; y++)
                {
                    for (int x = 0; x < labels.Cols; x++)
                    {
                        if (labels.At<int>(y, x) == segId)
                        {
                            var c = quantized.At<Vec3b>(y, x);
                            var key = (c[0], c[1], c[2]);
                            colorCounts[key] = colorCounts.GetValueOrDefault(key) + 1;
                        }
                    }
                }
                var mainColor = colorCounts.OrderByDescending(kvp => kvp.Value).First().Key;
                colors.Add(mainColor);  // Keep as tuple (byte, byte, byte)
            }

            LogPipeline($"[ExtractContours] Completed with {contours.Count} contours");
            return (contours, centroids, colors);
        }
        catch (Exception ex)
        {
            LogPipeline($"[ExtractContours] EXCEPTION: {ex.GetType().Name}: {ex.Message}");
            throw;
        }
    }

    private static void LogPipeline(string message)
    {
        try
        {
            var logPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "OrganicVectoriser",
                "pipeline.log"
            );
            var dir = Path.GetDirectoryName(logPath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }
            File.AppendAllText(logPath, $"[{DateTime.Now:HH:mm:ss.fff}] {message}\n");
        }
        catch
        {
            // Silent fail for logging
        }
    }
}

public sealed class BitmapInput
{
    public byte[] Pixels { get; }
    public int Width { get; }
    public int Height { get; }
    public int Stride { get; }
    public BitmapInput(byte[] pixels, int width, int height, int stride)
    {
        Pixels = pixels;
        Width = width;
        Height = height;
        Stride = stride;
    }
}
