namespace OrganicVectoriser.Models;

public class ParameterSet
{
    public double NoiseScale { get; set; } = 60.0;
    public double BlurSigma { get; set; } = 2.0;
    public double Compactness { get; set; } = 0.001;
    public int MaxColors { get; set; } = 8;

    public double BridgeDistance { get; set; } = 5.0;
    public double ColorTolerance { get; set; } = 10.0;
    public double ProximityThreshold { get; set; } = 50.0;
    public int FalloffRadius { get; set; } = 5;
    public double MaxCurvatureDegrees { get; set; } = 160.0;

    public int SmoothIterations { get; set; } = 3;
    public double SmoothAlpha { get; set; } = 0.3;

    public double BlobInflationAmount { get; set; } = 0.0;
    public double FarPointInflationFactor { get; set; } = 1.0;
    public bool InflationProportionalToStacking { get; set; } = true;

    public string StackingOrder { get; set; } = "area";
    public double SegmentationMultiplier { get; set; } = 1.0;

    // Droplet / painterly
    public string DropletStyle { get; set; } = "painterly"; // painterly|organic
    public int DropletDensity { get; set; } = 0;
    public double DropletMinDistance { get; set; } = 5.0;
    public double DropletMaxDistance { get; set; } = 15.0;
    public double DropletSizeMean { get; set; } = 3.0;
    public double DropletSizeStd { get; set; } = 1.0;
    public double DropletSpreadDegrees { get; set; } = 5.0;

    // Organic droplets
    public double DropletOrganicMinBrightness { get; set; } = 128.0;
    public int DropletOrganicDensity { get; set; } = 3;
    public double DropletOrganicStrength { get; set; } = 1.0;
    public double DropletOrganicJitter { get; set; } = 0.5;
    public double DropletOrganicElongation { get; set; } = 0.0;
    public double DropletOrganicPercentPerBlob { get; set; } = 100.0;

    // Painterly primitives
    public bool PainterlyUseSvgEllipses { get; set; } = false;
    public string PainterlySvgPrimitive { get; set; } = "ellipse"; // ellipse|rect
    public double DropletGlobalRotation { get; set; } = 0.0;
    public bool PainterlyRectHorizontal { get; set; } = false;

    public double SimplifyTolerance { get; set; } = 0.5;
}
