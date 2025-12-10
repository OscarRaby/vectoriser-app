using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using OrganicVectoriser.Models;
using SkiaSharp;

namespace OrganicVectoriser.Services;

public interface IRenderService
{
    ImageSource RenderPreview(PipelineResult result);
}

/// <summary>
/// Renders pipeline results to bitmap using SkiaSharp.
/// </summary>
public sealed class RenderService : IRenderService
{
    public ImageSource RenderPreview(PipelineResult result)
    {
        if (result == null)
            throw new ArgumentNullException(nameof(result));
            
        if (result.Width <= 0 || result.Height <= 0)
            throw new ArgumentException("Invalid result dimensions");

        if (result.Contours.Count == 0 && result.Droplets.Count == 0)
            throw new InvalidOperationException("Result has no contours or droplets to render");

        try
        {
            var info = new SKImageInfo(result.Width, result.Height, SKColorType.Bgra8888, SKAlphaType.Premul);
            using var surface = SKSurface.Create(info);
            var canvas = surface.Canvas;

            // White background
            canvas.Clear(SKColors.White);

            // Draw contours
            foreach (var contour in result.Contours)
            {
                if (contour.Points.Count < 3)
                    continue;

            using var path = new SKPath();
            path.MoveTo((float)contour.Points[0].X, (float)contour.Points[0].Y);
            for (int i = 1; i < contour.Points.Count; i++)
            {
                path.LineTo((float)contour.Points[i].X, (float)contour.Points[i].Y);
            }
            path.Close();

            using var paint = new SKPaint
            {
                Style = SKPaintStyle.Fill,
                Color = new SKColor(contour.FillColor.R, contour.FillColor.G, contour.FillColor.B),
                IsAntialias = true
            };
            canvas.DrawPath(path, paint);
        }

        // Draw droplets
        foreach (var droplet in result.Droplets)
        {
            var color = new SKColor(droplet.FillColor.R, droplet.FillColor.G, droplet.FillColor.B);
            using var paint = new SKPaint
            {
                Style = SKPaintStyle.Fill,
                Color = color,
                IsAntialias = true
            };

            switch (droplet.Kind)
            {
                case DropletKind.Polygon:
                    if (droplet.Polygon.Count >= 3)
                    {
                        using var path = new SKPath();
                        path.MoveTo((float)droplet.Polygon[0].X, (float)droplet.Polygon[0].Y);
                        for (int i = 1; i < droplet.Polygon.Count; i++)
                        {
                            path.LineTo((float)droplet.Polygon[i].X, (float)droplet.Polygon[i].Y);
                        }
                        path.Close();
                        canvas.DrawPath(path, paint);
                    }
                    break;

                case DropletKind.Ellipse:
                    DrawRotatedEllipse(canvas, paint, droplet);
                    break;

                case DropletKind.Rect:
                    DrawRotatedRect(canvas, paint, droplet);
                    break;
            }
        }

            // Convert to WPF BitmapSource
            using var image = surface.Snapshot();
            using var data = image.Encode(SKEncodedImageFormat.Png, 100);
            var bytes = data.ToArray();

            var bitmapImage = new BitmapImage();
            bitmapImage.BeginInit();
            bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
            bitmapImage.StreamSource = new System.IO.MemoryStream(bytes);
            bitmapImage.EndInit();
            bitmapImage.Freeze();

            return bitmapImage;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"RenderPreview error: {ex}");
            throw;
        }
    }

    private static void DrawRotatedEllipse(SKCanvas canvas, SKPaint paint, DropletInstance droplet)
    {
        canvas.Save();

        // Translate to center
        canvas.Translate((float)droplet.Cx, (float)droplet.Cy);

        // Rotate around origin
        canvas.RotateDegrees((float)droplet.AngleDegrees);

        // Draw ellipse at origin
        canvas.DrawOval(0, 0, (float)droplet.Rx, (float)droplet.Ry, paint);

        canvas.Restore();
    }

    private static void DrawRotatedRect(SKCanvas canvas, SKPaint paint, DropletInstance droplet)
    {
        canvas.Save();

        // Translate to center
        canvas.Translate((float)droplet.Cx, (float)droplet.Cy);

        // Rotate around origin
        canvas.RotateDegrees((float)droplet.AngleDegrees);

        // Draw rectangle centered at origin (Rx = width, Ry = height)
        var halfW = (float)droplet.Rx / 2.0f;
        var halfH = (float)droplet.Ry / 2.0f;
        canvas.DrawRect(-halfW, -halfH, (float)droplet.Rx, (float)droplet.Ry, paint);

        canvas.Restore();
    }
}
