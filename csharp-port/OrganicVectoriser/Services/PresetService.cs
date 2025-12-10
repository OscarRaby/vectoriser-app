using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using OrganicVectoriser.Models;

namespace OrganicVectoriser.Services;

public interface IPresetService
{
    Task<List<string>> GetPresetNamesAsync();
    Task SavePresetAsync(string name, ParameterSet parameters, ModifierFlags modifiers);
    Task<(ParameterSet parameters, ModifierFlags modifiers)?> LoadPresetAsync(string name);
    Task DeletePresetAsync(string name);
}

public sealed class PresetService : IPresetService
{
    private const string PresetsFileName = "vectoriser_presets.json";
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never
    };

    public async Task<List<string>> GetPresetNamesAsync()
    {
        if (!File.Exists(PresetsFileName))
            return new List<string>();

        try
        {
            var json = await File.ReadAllTextAsync(PresetsFileName).ConfigureAwait(false);
            var presets = JsonSerializer.Deserialize<Dictionary<string, PresetData>>(json, JsonOptions);
            return presets?.Keys.OrderBy(k => k).ToList() ?? new List<string>();
        }
        catch
        {
            return new List<string>();
        }
    }

    public async Task SavePresetAsync(string name, ParameterSet parameters, ModifierFlags modifiers)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Preset name cannot be empty.", nameof(name));

        var presets = await LoadAllPresetsAsync().ConfigureAwait(false);
        
        presets[name] = new PresetData
        {
            Parameters = CloneParameters(parameters),
            Modifiers = CloneModifiers(modifiers)
        };

        await SaveAllPresetsAsync(presets).ConfigureAwait(false);
    }

    public async Task<(ParameterSet parameters, ModifierFlags modifiers)?> LoadPresetAsync(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return null;

        var presets = await LoadAllPresetsAsync().ConfigureAwait(false);
        
        if (!presets.TryGetValue(name, out var preset))
            return null;

        return (CloneParameters(preset.Parameters), CloneModifiers(preset.Modifiers));
    }

    public async Task DeletePresetAsync(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return;

        var presets = await LoadAllPresetsAsync().ConfigureAwait(false);
        
        if (presets.Remove(name))
        {
            await SaveAllPresetsAsync(presets).ConfigureAwait(false);
        }
    }

    private async Task<Dictionary<string, PresetData>> LoadAllPresetsAsync()
    {
        if (!File.Exists(PresetsFileName))
            return new Dictionary<string, PresetData>();

        try
        {
            var json = await File.ReadAllTextAsync(PresetsFileName).ConfigureAwait(false);
            return JsonSerializer.Deserialize<Dictionary<string, PresetData>>(json, JsonOptions) 
                   ?? new Dictionary<string, PresetData>();
        }
        catch
        {
            return new Dictionary<string, PresetData>();
        }
    }

    private async Task SaveAllPresetsAsync(Dictionary<string, PresetData> presets)
    {
        var json = JsonSerializer.Serialize(presets, JsonOptions);
        await File.WriteAllTextAsync(PresetsFileName, json).ConfigureAwait(false);
    }

    private ParameterSet CloneParameters(ParameterSet source)
    {
        return new ParameterSet
        {
            NoiseScale = source.NoiseScale,
            BlurSigma = source.BlurSigma,
            Compactness = source.Compactness,
            MaxColors = source.MaxColors,
            BridgeDistance = source.BridgeDistance,
            ColorTolerance = source.ColorTolerance,
            ProximityThreshold = source.ProximityThreshold,
            FalloffRadius = source.FalloffRadius,
            MaxCurvatureDegrees = source.MaxCurvatureDegrees,
            SmoothIterations = source.SmoothIterations,
            SmoothAlpha = source.SmoothAlpha,
            BlobInflationAmount = source.BlobInflationAmount,
            FarPointInflationFactor = source.FarPointInflationFactor,
            InflationProportionalToStacking = source.InflationProportionalToStacking,
            StackingOrder = source.StackingOrder,
            SegmentationMultiplier = source.SegmentationMultiplier,
            DropletStyle = source.DropletStyle,
            DropletDensity = source.DropletDensity,
            DropletMinDistance = source.DropletMinDistance,
            DropletMaxDistance = source.DropletMaxDistance,
            DropletSizeMean = source.DropletSizeMean,
            DropletSizeStd = source.DropletSizeStd,
            DropletSpreadDegrees = source.DropletSpreadDegrees,
            DropletOrganicMinBrightness = source.DropletOrganicMinBrightness,
            DropletOrganicDensity = source.DropletOrganicDensity,
            DropletOrganicStrength = source.DropletOrganicStrength,
            DropletOrganicJitter = source.DropletOrganicJitter,
            DropletOrganicElongation = source.DropletOrganicElongation,
            DropletOrganicPercentPerBlob = source.DropletOrganicPercentPerBlob,
            PainterlyUseSvgEllipses = source.PainterlyUseSvgEllipses,
            PainterlySvgPrimitive = source.PainterlySvgPrimitive,
            DropletGlobalRotation = source.DropletGlobalRotation,
            PainterlyRectHorizontal = source.PainterlyRectHorizontal,
            SimplifyTolerance = source.SimplifyTolerance
        };
    }

    private ModifierFlags CloneModifiers(ModifierFlags source)
    {
        return new ModifierFlags
        {
            ColorQuantization = source.ColorQuantization,
            Bridging = source.Bridging,
            Smoothing = source.Smoothing,
            Inflation = source.Inflation,
            EnableVectorPreview = source.EnableVectorPreview
        };
    }

    private class PresetData
    {
        public ParameterSet Parameters { get; set; } = new();
        public ModifierFlags Modifiers { get; set; } = new();
    }
}
