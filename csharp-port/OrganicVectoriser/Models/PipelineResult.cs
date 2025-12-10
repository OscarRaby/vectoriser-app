using System.Collections.Generic;
using System.Windows.Media;

namespace OrganicVectoriser.Models;

public class PipelineResult
{
    public List<ContourData> Contours { get; } = new();
    public List<DropletInstance> Droplets { get; } = new();
    public int Width { get; set; }
    public int Height { get; set; }
}

public class ContourData
{
    public List<(double X, double Y)> Points { get; } = new();
    public (byte R, byte G, byte B) FillColor { get; set; } = (0, 0, 0);
}

public enum DropletKind
{
    Polygon,
    Ellipse,
    Rect
}

public class DropletInstance
{
    public DropletKind Kind { get; set; }
    public List<(double X, double Y)> Polygon { get; } = new();
    public double Cx { get; set; }
    public double Cy { get; set; }
    public double Rx { get; set; }
    public double Ry { get; set; }
    public double AngleDegrees { get; set; }
    public (byte R, byte G, byte B) FillColor { get; set; } = (0, 0, 0);
}

