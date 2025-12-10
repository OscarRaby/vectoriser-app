using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace OrganicVectoriser.Models;

public class ParameterSet : INotifyPropertyChanged
{
    private double _noiseScale = 60.0;
    public double NoiseScale
    {
        get => _noiseScale;
        set { if (_noiseScale != value) { _noiseScale = value; OnPropertyChanged(); } }
    }

    private double _blurSigma = 2.0;
    public double BlurSigma
    {
        get => _blurSigma;
        set { if (_blurSigma != value) { _blurSigma = value; OnPropertyChanged(); } }
    }

    private double _compactness = 0.001;
    public double Compactness
    {
        get => _compactness;
        set { if (_compactness != value) { _compactness = value; OnPropertyChanged(); } }
    }

    private int _maxColors = 8;
    public int MaxColors
    {
        get => _maxColors;
        set { if (_maxColors != value) { _maxColors = value; OnPropertyChanged(); } }
    }

    private double _bridgeDistance = 5.0;
    public double BridgeDistance
    {
        get => _bridgeDistance;
        set { if (_bridgeDistance != value) { _bridgeDistance = value; OnPropertyChanged(); } }
    }

    private double _colorTolerance = 10.0;
    public double ColorTolerance
    {
        get => _colorTolerance;
        set { if (_colorTolerance != value) { _colorTolerance = value; OnPropertyChanged(); } }
    }

    private double _proximityThreshold = 50.0;
    public double ProximityThreshold
    {
        get => _proximityThreshold;
        set { if (_proximityThreshold != value) { _proximityThreshold = value; OnPropertyChanged(); } }
    }

    private int _falloffRadius = 5;
    public int FalloffRadius
    {
        get => _falloffRadius;
        set { if (_falloffRadius != value) { _falloffRadius = value; OnPropertyChanged(); } }
    }

    private double _maxCurvatureDegrees = 160.0;
    public double MaxCurvatureDegrees
    {
        get => _maxCurvatureDegrees;
        set { if (_maxCurvatureDegrees != value) { _maxCurvatureDegrees = value; OnPropertyChanged(); } }
    }

    private int _smoothIterations = 3;
    public int SmoothIterations
    {
        get => _smoothIterations;
        set { if (_smoothIterations != value) { _smoothIterations = value; OnPropertyChanged(); } }
    }

    private double _smoothAlpha = 0.3;
    public double SmoothAlpha
    {
        get => _smoothAlpha;
        set { if (_smoothAlpha != value) { _smoothAlpha = value; OnPropertyChanged(); } }
    }

    private double _blobInflationAmount = 0.0;
    public double BlobInflationAmount
    {
        get => _blobInflationAmount;
        set { if (_blobInflationAmount != value) { _blobInflationAmount = value; OnPropertyChanged(); } }
    }

    private double _farPointInflationFactor = 1.0;
    public double FarPointInflationFactor
    {
        get => _farPointInflationFactor;
        set { if (_farPointInflationFactor != value) { _farPointInflationFactor = value; OnPropertyChanged(); } }
    }

    private bool _inflationProportionalToStacking = true;
    public bool InflationProportionalToStacking
    {
        get => _inflationProportionalToStacking;
        set { if (_inflationProportionalToStacking != value) { _inflationProportionalToStacking = value; OnPropertyChanged(); } }
    }

    private string _stackingOrder = "area";
    public string StackingOrder
    {
        get => _stackingOrder;
        set { if (_stackingOrder != value) { _stackingOrder = value; OnPropertyChanged(); } }
    }

    private double _segmentationMultiplier = 1.0;
    public double SegmentationMultiplier
    {
        get => _segmentationMultiplier;
        set { if (_segmentationMultiplier != value) { _segmentationMultiplier = value; OnPropertyChanged(); } }
    }

    private string _dropletStyle = "painterly";
    public string DropletStyle
    {
        get => _dropletStyle;
        set { if (_dropletStyle != value) { _dropletStyle = value; OnPropertyChanged(); } }
    }

    private int _dropletDensity = 0;
    public int DropletDensity
    {
        get => _dropletDensity;
        set { if (_dropletDensity != value) { _dropletDensity = value; OnPropertyChanged(); } }
    }

    private double _dropletMinDistance = 5.0;
    public double DropletMinDistance
    {
        get => _dropletMinDistance;
        set { if (_dropletMinDistance != value) { _dropletMinDistance = value; OnPropertyChanged(); } }
    }

    private double _dropletMaxDistance = 15.0;
    public double DropletMaxDistance
    {
        get => _dropletMaxDistance;
        set { if (_dropletMaxDistance != value) { _dropletMaxDistance = value; OnPropertyChanged(); } }
    }

    private double _dropletSizeMean = 3.0;
    public double DropletSizeMean
    {
        get => _dropletSizeMean;
        set { if (_dropletSizeMean != value) { _dropletSizeMean = value; OnPropertyChanged(); } }
    }

    private double _dropletSizeStd = 1.0;
    public double DropletSizeStd
    {
        get => _dropletSizeStd;
        set { if (_dropletSizeStd != value) { _dropletSizeStd = value; OnPropertyChanged(); } }
    }

    private double _dropletSpreadDegrees = 5.0;
    public double DropletSpreadDegrees
    {
        get => _dropletSpreadDegrees;
        set { if (_dropletSpreadDegrees != value) { _dropletSpreadDegrees = value; OnPropertyChanged(); } }
    }

    private double _dropletOrganicMinBrightness = 128.0;
    public double DropletOrganicMinBrightness
    {
        get => _dropletOrganicMinBrightness;
        set { if (_dropletOrganicMinBrightness != value) { _dropletOrganicMinBrightness = value; OnPropertyChanged(); } }
    }

    private int _dropletOrganicDensity = 3;
    public int DropletOrganicDensity
    {
        get => _dropletOrganicDensity;
        set { if (_dropletOrganicDensity != value) { _dropletOrganicDensity = value; OnPropertyChanged(); } }
    }

    private double _dropletOrganicStrength = 1.0;
    public double DropletOrganicStrength
    {
        get => _dropletOrganicStrength;
        set { if (_dropletOrganicStrength != value) { _dropletOrganicStrength = value; OnPropertyChanged(); } }
    }

    private double _dropletOrganicJitter = 0.5;
    public double DropletOrganicJitter
    {
        get => _dropletOrganicJitter;
        set { if (_dropletOrganicJitter != value) { _dropletOrganicJitter = value; OnPropertyChanged(); } }
    }

    private double _dropletOrganicElongation = 0.0;
    public double DropletOrganicElongation
    {
        get => _dropletOrganicElongation;
        set { if (_dropletOrganicElongation != value) { _dropletOrganicElongation = value; OnPropertyChanged(); } }
    }

    private double _dropletOrganicPercentPerBlob = 100.0;
    public double DropletOrganicPercentPerBlob
    {
        get => _dropletOrganicPercentPerBlob;
        set { if (_dropletOrganicPercentPerBlob != value) { _dropletOrganicPercentPerBlob = value; OnPropertyChanged(); } }
    }

    private bool _painterlyUseSvgEllipses = false;
    public bool PainterlyUseSvgEllipses
    {
        get => _painterlyUseSvgEllipses;
        set { if (_painterlyUseSvgEllipses != value) { _painterlyUseSvgEllipses = value; OnPropertyChanged(); } }
    }

    private string _painterlySvgPrimitive = "ellipse";
    public string PainterlySvgPrimitive
    {
        get => _painterlySvgPrimitive;
        set { if (_painterlySvgPrimitive != value) { _painterlySvgPrimitive = value; OnPropertyChanged(); } }
    }

    private double _dropletGlobalRotation = 0.0;
    public double DropletGlobalRotation
    {
        get => _dropletGlobalRotation;
        set { if (_dropletGlobalRotation != value) { _dropletGlobalRotation = value; OnPropertyChanged(); } }
    }

    private bool _painterlyRectHorizontal = false;
    public bool PainterlyRectHorizontal
    {
        get => _painterlyRectHorizontal;
        set { if (_painterlyRectHorizontal != value) { _painterlyRectHorizontal = value; OnPropertyChanged(); } }
    }

    private double _simplifyTolerance = 0.5;
    public double SimplifyTolerance
    {
        get => _simplifyTolerance;
        set { if (_simplifyTolerance != value) { _simplifyTolerance = value; OnPropertyChanged(); } }
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}

