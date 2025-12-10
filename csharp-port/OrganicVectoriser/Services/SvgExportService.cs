using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using OrganicVectoriser.Models;

namespace OrganicVectoriser.Services;

public interface ISvgExportService
{
    Task ExportAsync(string path, PipelineResult result, ParameterSet parameters);
}

/// <summary>
/// SVG export service that writes contours and droplets to an SVG file.
/// Matches Python svgwrite output format.
/// </summary>
public sealed class SvgExportService : ISvgExportService
{
    public Task ExportAsync(string path, PipelineResult result, ParameterSet parameters)
    {
        return Task.Run(() =>
        {
            using var writer = new StreamWriter(path, false, Encoding.UTF8);
            
            // XML declaration
            writer.WriteLine("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
            
            // SVG root element
            writer.WriteLine($"<svg width=\"{result.Width}px\" height=\"{result.Height}px\" viewBox=\"0 0 {result.Width} {result.Height}\" xmlns=\"http://www.w3.org/2000/svg\">");
            
            // Optional style definition for no stroke
            writer.WriteLine("  <defs>");
            writer.WriteLine("    <style>path { stroke: none; stroke-width: 0; }</style>");
            writer.WriteLine("  </defs>");
            
            // Write contours
            foreach (var contour in result.Contours)
            {
                if (contour.Points.Count == 0)
                    continue;

                var pathData = ContourToSvgPath(contour.Points, parameters.SimplifyTolerance);
                if (string.IsNullOrEmpty(pathData))
                    continue;

                var color = contour.FillColor;
                writer.WriteLine($"  <path d=\"{pathData}\" fill=\"rgb({color.R},{color.G},{color.B})\" />");
            }
            
            // Write droplets
            foreach (var droplet in result.Droplets)
            {
                var c = droplet.FillColor;
                var colorStr = $"rgb({c.R},{c.G},{c.B})";

                switch (droplet.Kind)
                {
                    case DropletKind.Polygon:
                        var pathData = ContourToSvgPath(droplet.Polygon, parameters.SimplifyTolerance);
                        if (!string.IsNullOrEmpty(pathData))
                            writer.WriteLine($"  <path d=\"{pathData}\" fill=\"{colorStr}\" />");
                        break;

                    case DropletKind.Ellipse:
                        var cx = FormatCoord(droplet.Cx);
                        var cy = FormatCoord(droplet.Cy);
                        var rx = FormatCoord(droplet.Rx);
                        var ry = FormatCoord(droplet.Ry);
                        var angle = FormatCoord(droplet.AngleDegrees);
                        
                        if (Math.Abs(droplet.AngleDegrees) < 0.01)
                        {
                            writer.WriteLine($"  <ellipse cx=\"{cx}\" cy=\"{cy}\" rx=\"{rx}\" ry=\"{ry}\" fill=\"{colorStr}\" />");
                        }
                        else
                        {
                            writer.WriteLine($"  <ellipse cx=\"{cx}\" cy=\"{cy}\" rx=\"{rx}\" ry=\"{ry}\" fill=\"{colorStr}\" transform=\"rotate({angle} {cx} {cy})\" />");
                        }
                        break;

                    case DropletKind.Rect:
                        // Rx and Ry store width and height respectively
                        var rcx = FormatCoord(droplet.Cx);
                        var rcy = FormatCoord(droplet.Cy);
                        var w = FormatCoord(droplet.Rx);
                        var h = FormatCoord(droplet.Ry);
                        var rangle = FormatCoord(droplet.AngleDegrees);
                        
                        // Calculate top-left corner (rect uses x,y for insertion point)
                        var x = FormatCoord(droplet.Cx - droplet.Rx / 2.0);
                        var y = FormatCoord(droplet.Cy - droplet.Ry / 2.0);
                        
                        if (Math.Abs(droplet.AngleDegrees) < 0.01)
                        {
                            writer.WriteLine($"  <rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" fill=\"{colorStr}\" />");
                        }
                        else
                        {
                            writer.WriteLine($"  <rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" fill=\"{colorStr}\" transform=\"rotate({rangle} {rcx} {rcy})\" />");
                        }
                        break;
                }
            }
            
            writer.WriteLine("</svg>");
        });
    }

    /// <summary>
    /// Convert a list of points to an SVG path string.
    /// Points are in (X, Y) format where X = column, Y = row.
    /// </summary>
    private static string ContourToSvgPath(System.Collections.Generic.List<(double X, double Y)> points, double simplifyTol)
    {
        if (points.Count == 0)
            return string.Empty;

        var sb = new StringBuilder();
        sb.Append("M ");

        // First point
        sb.Append(FormatCoord(points[0].X));
        sb.Append(',');
        sb.Append(FormatCoord(points[0].Y));

        // Remaining points
        for (int i = 1; i < points.Count; i++)
        {
            sb.Append(" L ");
            sb.Append(FormatCoord(points[i].X));
            sb.Append(',');
            sb.Append(FormatCoord(points[i].Y));
        }

        // Close path
        sb.Append(" Z");

        return sb.ToString();
    }

    /// <summary>
    /// Format a coordinate value with one decimal place.
    /// Uses invariant culture to ensure decimal point (not comma).
    /// </summary>
    private static string FormatCoord(double value)
    {
        return Math.Round(value, 1).ToString("F1", CultureInfo.InvariantCulture);
    }
}
