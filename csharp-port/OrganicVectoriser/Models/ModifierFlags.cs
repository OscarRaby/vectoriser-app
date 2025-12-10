namespace OrganicVectoriser.Models;

public class ModifierFlags
{
    public bool ColorQuantization { get; set; } = true;
    public bool Bridging { get; set; } = true;
    public bool Smoothing { get; set; } = true;
    public bool Inflation { get; set; } = true;
    public bool EnableVectorPreview { get; set; } = true;
}
