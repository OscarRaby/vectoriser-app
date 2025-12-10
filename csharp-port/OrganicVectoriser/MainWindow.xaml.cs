using System.Windows;
using OrganicVectoriser.ViewModels;

namespace OrganicVectoriser;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        
        // Fix WPF binding initialization issue where two-way slider bindings 
        // might cause the Slider's default value (0) to overwrite the source's default value
        this.Loaded += (s, e) =>
        {
            if (DataContext is MainViewModel vm)
            {
                // Trigger a refresh of all parameter properties by reassigning them
                // This causes INotifyPropertyChanged to fire and updates any UI that hasn't bound correctly
                vm.Parameters.NoiseScale = vm.Parameters.NoiseScale;
                vm.Parameters.BlurSigma = vm.Parameters.BlurSigma;
                vm.Parameters.Compactness = vm.Parameters.Compactness;
                vm.Parameters.MaxColors = vm.Parameters.MaxColors;
                vm.Parameters.SegmentationMultiplier = vm.Parameters.SegmentationMultiplier;
                vm.Parameters.BridgeDistance = vm.Parameters.BridgeDistance;
                vm.Parameters.ColorTolerance = vm.Parameters.ColorTolerance;
                vm.Parameters.ProximityThreshold = vm.Parameters.ProximityThreshold;
            }
        };
    }
}

