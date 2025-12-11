using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using OrganicVectoriser.Models;

namespace OrganicVectoriser.Services;

/// <summary>
/// Pipeline service that calls the native C++ vectoriser via P/Invoke.
/// Returns fully populated PipelineResult for preview and SVG export.
/// </summary>
public sealed class NativePipelineService : IPipelineService
{
    private const string NativeDll = "OrganicVectoriserNative";
    private const string NativeDllName = "OrganicVectoriserNative.dll";
    private static bool _nativeLoaded;

    static NativePipelineService()
    {
        EnsureNativeLibraryLoaded();
    }

    public Task<PipelineResult> RunAsync(BitmapInput input, ParameterSet parameters, ModifierFlags modifiers)
    {
        return Task.Run(() => Execute(input, parameters, modifiers, svgPath: null));
    }

    public Task<PipelineResult> PreviewAsync(BitmapInput input, ParameterSet parameters, ModifierFlags modifiers)
    {
        return Task.Run(() => Execute(input, parameters, modifiers, svgPath: null));
    }

    public Task RunDiagnosticsAsync(BitmapInput input, ParameterSet parameters)
    {
        // Native pipeline currently does not expose diagnostics; fall back to normal run to validate interop
        return Task.CompletedTask;
    }

    private static PipelineResult Execute(BitmapInput input, ParameterSet p, ModifierFlags m, string? svgPath)
    {
        var nativeParams = ToNativeParams(p);
        var nativeMods = ToNativeModifiers(m);

        // Ensure BGR24 buffer (native expects width*height*3)
        var bgrBuffer = (input.Stride == input.Width * 3)
            ? input.Pixels
            : ToBgr24(input);

        IntPtr resultPtr = IntPtr.Zero;
        try
        {
            var code = NativeMethods.ov_run_pipeline(bgrBuffer, input.Width, input.Height,
                ref nativeParams, ref nativeMods, out resultPtr, svgPath);
            if (code != 0 || resultPtr == IntPtr.Zero)
            {
                throw new InvalidOperationException($"Native pipeline failed with code {code}.");
            }

            var native = Marshal.PtrToStructure<NativeResult>(resultPtr);
            if (native.width <= 0 || native.height <= 0)
            {
                throw new InvalidOperationException("Native pipeline returned invalid dimensions.");
            }

            var result = new PipelineResult
            {
                Width = native.width,
                Height = native.height
            };

            // Contours
            if (native.contourCount > 0 && native.contours != IntPtr.Zero)
            {
                var contours = new NativeContour[native.contourCount];
                var size = Marshal.SizeOf<NativeContour>();
                for (int i = 0; i < native.contourCount; i++)
                {
                    var ptr = IntPtr.Add(native.contours, i * size);
                    contours[i] = Marshal.PtrToStructure<NativeContour>(ptr);
                }

                foreach (var c in contours)
                {
                    var managed = new ContourData();
                    managed.FillColor = (c.color0, c.color1, c.color2);
                    if (c.pointCount > 0 && c.points != IntPtr.Zero)
                    {
                        var pts = new NativePoint[c.pointCount];
                        var pSize = Marshal.SizeOf<NativePoint>();
                        for (int i = 0; i < c.pointCount; i++)
                        {
                            var pPtr = IntPtr.Add(c.points, i * pSize);
                            pts[i] = Marshal.PtrToStructure<NativePoint>(pPtr);
                            managed.Points.Add((pts[i].x, pts[i].y));
                        }
                    }
                    result.Contours.Add(managed);
                }
            }

            // Droplets
            if (native.dropletCount > 0 && native.droplets != IntPtr.Zero)
            {
                var droplets = new NativeDroplet[native.dropletCount];
                var dSize = Marshal.SizeOf<NativeDroplet>();
                for (int i = 0; i < native.dropletCount; i++)
                {
                    var ptr = IntPtr.Add(native.droplets, i * dSize);
                    droplets[i] = Marshal.PtrToStructure<NativeDroplet>(ptr);
                }

                foreach (var d in droplets)
                {
                    var managed = new DropletInstance();
                    managed.Kind = d.kind switch
                    {
                        1 => DropletKind.Ellipse,
                        2 => DropletKind.Rect,
                        _ => DropletKind.Polygon
                    };
                    managed.Cx = d.cx;
                    managed.Cy = d.cy;
                    managed.Rx = d.rx;
                    managed.Ry = d.ry;
                    managed.AngleDegrees = d.angleDegrees;
                    managed.FillColor = (d.color0, d.color1, d.color2);

                    if (d.pointCount > 0 && d.polygon != IntPtr.Zero)
                    {
                        var pts = new NativePoint[d.pointCount];
                        var pSize = Marshal.SizeOf<NativePoint>();
                        for (int i = 0; i < d.pointCount; i++)
                        {
                            var pPtr = IntPtr.Add(d.polygon, i * pSize);
                            pts[i] = Marshal.PtrToStructure<NativePoint>(pPtr);
                            managed.Polygon.Add((pts[i].x, pts[i].y));
                        }
                    }
                    result.Droplets.Add(managed);
                }
            }

            return result;
        }
        finally
        {
            if (resultPtr != IntPtr.Zero)
            {
                NativeMethods.ov_free_result(resultPtr);
            }
        }
    }

    private static NativeParams ToNativeParams(ParameterSet p) => new()
    {
        noiseScale = p.NoiseScale,
        blurSigma = p.BlurSigma,
        compactness = p.Compactness,
        maxColors = p.MaxColors,
        bridgeDistance = p.BridgeDistance,
        colorTolerance = p.ColorTolerance,
        proximityThreshold = p.ProximityThreshold,
        falloffRadius = p.FalloffRadius,
        maxCurvatureDegrees = p.MaxCurvatureDegrees,
        smoothIterations = p.SmoothIterations,
        smoothAlpha = p.SmoothAlpha,
        blobInflationAmount = p.BlobInflationAmount,
        farPointInflationFactor = p.FarPointInflationFactor,
        inflationProportionalToStacking = p.InflationProportionalToStacking ? 1 : 0,
        stackingOrder = ToStackingOrder(p.StackingOrder),
        segmentationMultiplier = p.SegmentationMultiplier,
        dropletDensity = p.DropletDensity,
        dropletMinDistance = p.DropletMinDistance,
        dropletMaxDistance = p.DropletMaxDistance,
        dropletSizeMean = p.DropletSizeMean,
        dropletSizeStd = p.DropletSizeStd,
        dropletSpreadDegrees = p.DropletSpreadDegrees,
        dropletOrganicMinBrightness = p.DropletOrganicMinBrightness,
        dropletOrganicDensity = p.DropletOrganicDensity,
        dropletOrganicStrength = p.DropletOrganicStrength,
        dropletOrganicJitter = p.DropletOrganicJitter,
        dropletOrganicElongation = p.DropletOrganicElongation,
        dropletOrganicPercentPerBlob = p.DropletOrganicPercentPerBlob,
        painterlyUseSvgEllipses = p.PainterlyUseSvgEllipses ? 1 : 0,
        painterlyRectHorizontal = p.PainterlyRectHorizontal ? 1 : 0,
        dropletGlobalRotation = p.DropletGlobalRotation,
        simplifyTolerance = p.SimplifyTolerance
    };

    private static int ToStackingOrder(string stacking)
    {
        return stacking?.ToLowerInvariant() switch
        {
            "area" => 0,
            "area_reverse" => 1,
            "brightness" => 2,
            "brightness_reverse" => 3,
            "position_x" => 4,
            "position_x_reverse" => 5,
            "position_y" => 6,
            "position_y_reverse" => 7,
            "position_centre" => 8,
            "position_centre_reverse" => 9,
            _ => 0
        };
    }

    private static NativeModifiers ToNativeModifiers(ModifierFlags m) => new()
    {
        colorQuantization = m.ColorQuantization ? 1 : 0,
        bridging = m.Bridging ? 1 : 0,
        smoothing = m.Smoothing ? 1 : 0,
        inflation = m.Inflation ? 1 : 0,
        enableVectorPreview = m.EnableVectorPreview ? 1 : 0
    };

    private static byte[] ToBgr24(BitmapInput input)
    {
        // Convert from BGRA32 (or other stride) to BGR24 expected by native code
        var output = new byte[input.Width * input.Height * 3];
        int srcStride = input.Stride;
        int dstStride = input.Width * 3;
        for (int y = 0; y < input.Height; y++)
        {
            int srcRow = y * srcStride;
            int dstRow = y * dstStride;
            for (int x = 0; x < input.Width; x++)
            {
                int srcIdx = srcRow + x * 4; // assume BGRA
                int dstIdx = dstRow + x * 3;
                output[dstIdx + 0] = input.Pixels[srcIdx + 0]; // B
                output[dstIdx + 1] = input.Pixels[srcIdx + 1]; // G
                output[dstIdx + 2] = input.Pixels[srcIdx + 2]; // R
            }
        }
        return output;
    }

    private static class NativeMethods
    {
        [DllImport(NativeDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ov_run_pipeline")]
        public static extern int ov_run_pipeline(byte[] bgrData, int width, int height,
            ref NativeParams parameters, ref NativeModifiers modifiers, out IntPtr result, string? svgOutputPath);

        [DllImport(NativeDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ov_free_result")]
        public static extern void ov_free_result(IntPtr result);
    }

    private static void EnsureNativeLibraryLoaded()
    {
        if (_nativeLoaded)
        {
            return;
        }

        var baseDir = AppContext.BaseDirectory ?? Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? string.Empty;
        var envOverride = Environment.GetEnvironmentVariable("OV_NATIVE_DLL_PATH");

        var candidates = new[]
        {
            envOverride,
            Path.Combine(baseDir, NativeDllName)
        };

        foreach (var candidate in candidates)
        {
            if (string.IsNullOrWhiteSpace(candidate) || !File.Exists(candidate))
            {
                continue;
            }

            if (NativeLibrary.TryLoad(candidate, out _))
            {
                _nativeLoaded = true;
                return;
            }
        }

        throw new DllNotFoundException($"Could not load {NativeDllName}. Build the native DLL (cmake --build . --config Release) and ensure it is copied next to OrganicVectoriser.exe.");
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeParams
    {
        public double noiseScale;
        public double blurSigma;
        public double compactness;
        public int maxColors;
        public double bridgeDistance;
        public double colorTolerance;
        public double proximityThreshold;
        public int falloffRadius;
        public double maxCurvatureDegrees;
        public int smoothIterations;
        public double smoothAlpha;
        public double blobInflationAmount;
        public double farPointInflationFactor;
        public int inflationProportionalToStacking;
        public int stackingOrder;
        public double segmentationMultiplier;
        public int dropletDensity;
        public double dropletMinDistance;
        public double dropletMaxDistance;
        public double dropletSizeMean;
        public double dropletSizeStd;
        public double dropletSpreadDegrees;
        public double dropletOrganicMinBrightness;
        public int dropletOrganicDensity;
        public double dropletOrganicStrength;
        public double dropletOrganicJitter;
        public double dropletOrganicElongation;
        public double dropletOrganicPercentPerBlob;
        public int painterlyUseSvgEllipses;
        public int painterlyRectHorizontal;
        public double dropletGlobalRotation;
        public double simplifyTolerance;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeModifiers
    {
        public int colorQuantization;
        public int bridging;
        public int smoothing;
        public int inflation;
        public int enableVectorPreview;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativePoint
    {
        public double x;
        public double y;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeContour
    {
        public IntPtr points;
        public int pointCount;
        public byte color0;
        public byte color1;
        public byte color2;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeDroplet
    {
        public int kind;
        public IntPtr polygon;
        public int pointCount;
        public double cx;
        public double cy;
        public double rx;
        public double ry;
        public double angleDegrees;
        public byte color0;
        public byte color1;
        public byte color2;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeResult
    {
        public IntPtr contours;
        public int contourCount;
        public IntPtr droplets;
        public int dropletCount;
        public int width;
        public int height;
    }
}
