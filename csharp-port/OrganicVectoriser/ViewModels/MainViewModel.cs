using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using OrganicVectoriser.Models;
using OrganicVectoriser.Services;
using OrganicVectoriser.ViewModels.Commands;

namespace OrganicVectoriser.ViewModels;

public sealed class MainViewModel : BaseViewModel
{
    private readonly IImageService _imageService = new ImageService();
    private readonly IPipelineService _pipelineService = new PipelineService();
    private readonly ISvgExportService _svgExportService = new SvgExportService();
    private readonly IRenderService _renderService = new RenderService();
    private readonly IPresetService _presetService = new PresetService();

    public ParameterSet Parameters { get; } = new();
    public ModifierFlags Modifiers { get; } = new();

    private string _svgOutputPath = "out.svg";
    public string SvgOutputPath
    {
        get => _svgOutputPath;
        set => SetProperty(ref _svgOutputPath, value);
    }

    private double _progress;
    public double Progress
    {
        get => _progress;
        set => SetProperty(ref _progress, value);
    }

    private string _status = "Ready.";
    public string Status
    {
        get => _status;
        set => SetProperty(ref _status, value);
    }

    private ImageSource? _previewImage;
    public ImageSource? PreviewImage
    {
        get => _previewImage;
        set
        {
            if (SetProperty(ref _previewImage, value))
            {
                RaisePropertyChanged(nameof(IsPreviewAvailable));
            }
        }
    }

    public bool IsPreviewAvailable => PreviewImage is not null;

    public ObservableCollection<string> AvailablePresets { get; } = new();
    private string? _selectedPreset;

    public string DropletStyle
    {
        get => Parameters.DropletStyle;
        set
        {
            if (Parameters.DropletStyle != value)
            {
                Parameters.DropletStyle = value;
                RaisePropertyChanged();
            }
        }
    }
    public string? SelectedPreset
    {
        get => _selectedPreset;
        set => SetProperty(ref _selectedPreset, value);
    }

    private string _currentPresetName = string.Empty;
    public string CurrentPresetName
    {
        get => _currentPresetName;
        set => SetProperty(ref _currentPresetName, value);
    }

    public RelayCommand LoadImageCommand { get; }
    public AsyncRelayCommand PreviewCommand { get; }
    public AsyncRelayCommand RunCommand { get; }
    public RelayCommand SavePresetCommand { get; }
    public RelayCommand LoadPresetCommand { get; }
    public RelayCommand DeletePresetCommand { get; }
    public RelayCommand UndoCommand { get; }
    public RelayCommand RedoCommand { get; }
    public AsyncRelayCommand DiagnosticsCommand { get; }

    private BitmapSource? _currentBitmap;

    public MainViewModel()
    {
        LoadImageCommand = new RelayCommand(LoadImage);
        PreviewCommand = new AsyncRelayCommand(PreviewAsync, () => _currentBitmap is not null);
        RunCommand = new AsyncRelayCommand(RunAsync, () => _currentBitmap is not null);
        SavePresetCommand = new RelayCommand(SavePreset);
        LoadPresetCommand = new RelayCommand(LoadPreset);
        DeletePresetCommand = new RelayCommand(DeletePreset);
        UndoCommand = new RelayCommand(() => { /* TODO */ });
        RedoCommand = new RelayCommand(() => { /* TODO */ });
        DiagnosticsCommand = new AsyncRelayCommand(RunDiagnosticsAsync, () => _currentBitmap is not null);

        // Load available presets asynchronously
        _ = RefreshPresetsAsync();
    }

    private void LoadImage()
    {
        var dlg = new OpenFileDialog
        {
            Filter = "Image files|*.png;*.jpg;*.jpeg;*.bmp|All files|*.*"
        };
        if (dlg.ShowDialog() != true) return;

        Status = "Loading image...";
        Task.Run(async () =>
        {
            try
            {
                var bmp = await _imageService.LoadAsync(dlg.FileName).ConfigureAwait(false);
                _currentBitmap = bmp;
                var resized = await _imageService.ResizeAsync(bmp, 800, 600).ConfigureAwait(false);
                resized.Freeze();
                await App.Current.Dispatcher.InvokeAsync(() =>
                {
                    PreviewImage = resized;
                    Status = "Image loaded.";
                    Progress = 0;
                    
                    // Notify commands that image is now available
                    PreviewCommand.RaiseCanExecuteChanged();
                    RunCommand.RaiseCanExecuteChanged();
                    DiagnosticsCommand.RaiseCanExecuteChanged();
                });
            }
            catch (Exception ex)
            {
                await App.Current.Dispatcher.InvokeAsync(() => Status = $"Load failed: {ex.Message}");
            }
        });
    }

    private async Task PreviewAsync()
    {
        if (_currentBitmap is null) return;
        try
        {
            Progress = 10;
            Status = "Running preview...";
            var input = ExtractBitmap(_currentBitmap);
            var result = await _pipelineService.PreviewAsync(input, Parameters, Modifiers).ConfigureAwait(false);
            
            // Render preview from result
            await App.Current.Dispatcher.InvokeAsync(() =>
            {
                try
                {
                    PreviewImage = _renderService.RenderPreview(result);
                    Status = "Preview complete.";
                }
                catch (Exception ex)
                {
                    Status = $"Preview rendering failed: {ex.Message}";
                    System.Diagnostics.Debug.WriteLine($"Preview render error: {ex}");
                }
                Progress = 0;
            });
        }
        catch (Exception ex)
        {
            Status = $"Preview failed: {ex.Message}";
            System.Diagnostics.Debug.WriteLine($"Preview error: {ex}");
            
            // Show error dialog for critical errors
            if (ex.InnerException?.Message.Contains("NativeMethods") ?? false)
            {
                await App.Current.Dispatcher.InvokeAsync(() =>
                {
                    System.Windows.MessageBox.Show(
                        $"Preview failed:\n\n{ex.Message}\n\nInner exception:\n{ex.InnerException?.Message}\n\nStack trace:\n{ex.StackTrace}",
                        "Preview Error",
                        System.Windows.MessageBoxButton.OK,
                        System.Windows.MessageBoxImage.Error
                    );
                });
            }
            Progress = 0;
        }
    }

    private async Task RunAsync()
    {
        if (_currentBitmap is null) return;
        try
        {
            Progress = 5;
            Status = "Running vectoriser...";
            var input = ExtractBitmap(_currentBitmap);
            var result = await _pipelineService.RunAsync(input, Parameters, Modifiers).ConfigureAwait(false);
            Progress = 90;
            await _svgExportService.ExportAsync(SvgOutputPath, result, Parameters).ConfigureAwait(false);
            Status = $"Saved SVG to {Path.GetFullPath(SvgOutputPath)}";
            Progress = 0;
        }
        catch (Exception ex)
        {
            Status = $"Run failed: {ex.Message}";
            System.Diagnostics.Debug.WriteLine($"Run error: {ex}");
            
            // Show error dialog for critical errors
            if (ex.InnerException?.Message.Contains("NativeMethods") ?? false)
            {
                await App.Current.Dispatcher.InvokeAsync(() =>
                {
                    System.Windows.MessageBox.Show(
                        $"Run failed:\n\n{ex.Message}\n\nInner exception:\n{ex.InnerException?.Message}\n\nStack trace:\n{ex.StackTrace}",
                        "Run Error",
                        System.Windows.MessageBoxButton.OK,
                        System.Windows.MessageBoxImage.Error
                    );
                });
            }
            Progress = 0;
        }
    }

    private async Task RunDiagnosticsAsync()
    {
        if (_currentBitmap is null) return;
        try
        {
            Status = "Running diagnostics...";
            var input = ExtractBitmap(_currentBitmap);
            await _pipelineService.RunDiagnosticsAsync(input, Parameters).ConfigureAwait(false);
            
            await App.Current.Dispatcher.InvokeAsync(() =>
            {
                var appDataPath = System.Environment.GetFolderPath(System.Environment.SpecialFolder.ApplicationData);
                var diagPath = System.IO.Path.Combine(appDataPath, "OrganicVectoriser", "diagnostics");
                var diagLog = System.IO.Path.Combine(diagPath, "diagnostics.log");
                
                Status = $"Diagnostics complete! Check {diagPath}";
                
                var message = $"Diagnostic results saved to:\n{diagPath}\n\n" +
                             $"Files created:\n" +
                             $"• 01_noise_*.png\n" +
                             $"• 02_elevation_*.png\n" +
                             $"• 03a_distance_transform.png\n" +
                             $"• 03b_local_maxima.png\n" +
                             $"• 04_watershed_result.png\n" +
                             $"• diagnostics.log (detailed output)\n\n" +
                             $"Open diagnostics.log to see all statistics!";
                
                System.Windows.MessageBox.Show(message, "Diagnostics Complete", 
                    System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Information);
            });
        }
        catch (Exception ex)
        {
            Status = $"Diagnostics failed: {ex.Message}";
            System.Diagnostics.Debug.WriteLine($"Diagnostics error: {ex}");
        }
    }

    private static BitmapInput ExtractBitmap(BitmapSource source)
    {
        var width = source.PixelWidth;
        var height = source.PixelHeight;
        var stride = (width * source.Format.BitsPerPixel + 7) / 8;
        var buffer = new byte[stride * height];
        source.CopyPixels(buffer, stride, 0);
        return new BitmapInput(buffer, width, height, stride);
    }

    private void SavePreset()
    {
        _ = SavePresetAsync();
    }

    private async Task SavePresetAsync()
    {
        var name = CurrentPresetName?.Trim();
        if (string.IsNullOrWhiteSpace(name))
        {
            Status = "Enter a preset name.";
            return;
        }

        try
        {
            await _presetService.SavePresetAsync(name, Parameters, Modifiers).ConfigureAwait(false);
            await RefreshPresetsAsync().ConfigureAwait(false);
            await App.Current.Dispatcher.InvokeAsync(() =>
            {
                Status = $"Preset '{name}' saved.";
                SelectedPreset = name;
            });
        }
        catch (Exception ex)
        {
            Status = $"Failed to save preset: {ex.Message}";
        }
    }

    private void LoadPreset()
    {
        _ = LoadPresetAsync();
    }

    private async Task LoadPresetAsync()
    {
        var name = SelectedPreset;
        if (string.IsNullOrWhiteSpace(name))
        {
            Status = "Select a preset to load.";
            return;
        }

        try
        {
            var preset = await _presetService.LoadPresetAsync(name).ConfigureAwait(false);
            if (preset is null)
            {
                Status = $"Preset '{name}' not found.";
                return;
            }

            await App.Current.Dispatcher.InvokeAsync(() =>
            {
                // Copy all parameter values
                Parameters.NoiseScale = preset.Value.parameters.NoiseScale;
                Parameters.BlurSigma = preset.Value.parameters.BlurSigma;
                Parameters.Compactness = preset.Value.parameters.Compactness;
                Parameters.MaxColors = preset.Value.parameters.MaxColors;
                Parameters.BridgeDistance = preset.Value.parameters.BridgeDistance;
                Parameters.ColorTolerance = preset.Value.parameters.ColorTolerance;
                Parameters.ProximityThreshold = preset.Value.parameters.ProximityThreshold;
                Parameters.FalloffRadius = preset.Value.parameters.FalloffRadius;
                Parameters.MaxCurvatureDegrees = preset.Value.parameters.MaxCurvatureDegrees;
                Parameters.SmoothIterations = preset.Value.parameters.SmoothIterations;
                Parameters.SmoothAlpha = preset.Value.parameters.SmoothAlpha;
                Parameters.BlobInflationAmount = preset.Value.parameters.BlobInflationAmount;
                Parameters.FarPointInflationFactor = preset.Value.parameters.FarPointInflationFactor;
                Parameters.InflationProportionalToStacking = preset.Value.parameters.InflationProportionalToStacking;
                Parameters.StackingOrder = preset.Value.parameters.StackingOrder;
                Parameters.SegmentationMultiplier = preset.Value.parameters.SegmentationMultiplier;
                Parameters.DropletStyle = preset.Value.parameters.DropletStyle;
                Parameters.DropletDensity = preset.Value.parameters.DropletDensity;
                Parameters.DropletMinDistance = preset.Value.parameters.DropletMinDistance;
                Parameters.DropletMaxDistance = preset.Value.parameters.DropletMaxDistance;
                Parameters.DropletSizeMean = preset.Value.parameters.DropletSizeMean;
                Parameters.DropletSizeStd = preset.Value.parameters.DropletSizeStd;
                Parameters.DropletSpreadDegrees = preset.Value.parameters.DropletSpreadDegrees;
                Parameters.DropletOrganicMinBrightness = preset.Value.parameters.DropletOrganicMinBrightness;
                Parameters.DropletOrganicDensity = preset.Value.parameters.DropletOrganicDensity;
                Parameters.DropletOrganicStrength = preset.Value.parameters.DropletOrganicStrength;
                Parameters.DropletOrganicJitter = preset.Value.parameters.DropletOrganicJitter;
                Parameters.DropletOrganicElongation = preset.Value.parameters.DropletOrganicElongation;
                Parameters.DropletOrganicPercentPerBlob = preset.Value.parameters.DropletOrganicPercentPerBlob;
                Parameters.PainterlyUseSvgEllipses = preset.Value.parameters.PainterlyUseSvgEllipses;
                Parameters.PainterlySvgPrimitive = preset.Value.parameters.PainterlySvgPrimitive;
                Parameters.DropletGlobalRotation = preset.Value.parameters.DropletGlobalRotation;
                Parameters.PainterlyRectHorizontal = preset.Value.parameters.PainterlyRectHorizontal;
                Parameters.SimplifyTolerance = preset.Value.parameters.SimplifyTolerance;

                // Copy modifier flags
                Modifiers.ColorQuantization = preset.Value.modifiers.ColorQuantization;
                Modifiers.Bridging = preset.Value.modifiers.Bridging;
                Modifiers.Smoothing = preset.Value.modifiers.Smoothing;
                Modifiers.Inflation = preset.Value.modifiers.Inflation;
                Modifiers.EnableVectorPreview = preset.Value.modifiers.EnableVectorPreview;

                Status = $"Preset '{name}' loaded.";
            });
        }
        catch (Exception ex)
        {
            Status = $"Failed to load preset: {ex.Message}";
        }
    }

    private void DeletePreset()
    {
        _ = DeletePresetAsync();
    }

    private async Task DeletePresetAsync()
    {
        var name = SelectedPreset;
        if (string.IsNullOrWhiteSpace(name))
        {
            Status = "Select a preset to delete.";
            return;
        }

        try
        {
            await _presetService.DeletePresetAsync(name).ConfigureAwait(false);
            await RefreshPresetsAsync().ConfigureAwait(false);
            await App.Current.Dispatcher.InvokeAsync(() =>
            {
                Status = $"Preset '{name}' deleted.";
                SelectedPreset = null;
            });
        }
        catch (Exception ex)
        {
            Status = $"Failed to delete preset: {ex.Message}";
        }
    }

    private async Task RefreshPresetsAsync()
    {
        try
        {
            var names = await _presetService.GetPresetNamesAsync().ConfigureAwait(false);
            await App.Current.Dispatcher.InvokeAsync(() =>
            {
                AvailablePresets.Clear();
                foreach (var name in names)
                {
                    AvailablePresets.Add(name);
                }
            });
        }
        catch
        {
            // Ignore errors when loading presets list
        }
    }
}
