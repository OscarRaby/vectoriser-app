using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace OrganicVectoriser.Services;

public interface IImageService
{
    Task<BitmapSource> LoadAsync(string path);
    Task<BitmapSource> ResizeAsync(BitmapSource source, int maxWidth, int maxHeight);
}

public sealed class ImageService : IImageService
{
    public Task<BitmapSource> LoadAsync(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException("Image not found", path);
        return Task.Run(() =>
        {
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.UriSource = new Uri(path, UriKind.Absolute);
            bmp.EndInit();
            
            // Convert to a standard format (RGB24) to ensure compatibility
            if (bmp.Format != System.Windows.Media.PixelFormats.Bgr24 && 
                bmp.Format != System.Windows.Media.PixelFormats.Rgb24)
            {
                var converted = new FormatConvertedBitmap(bmp, System.Windows.Media.PixelFormats.Bgr24, null, 0);
                converted.Freeze();
                return (BitmapSource)converted;
            }
            
            bmp.Freeze();
            return (BitmapSource)bmp;
        });
    }

    public Task<BitmapSource> ResizeAsync(BitmapSource source, int maxWidth, int maxHeight)
    {
        return Task.Run(() =>
        {
            var scale = Math.Min((double)maxWidth / source.PixelWidth, (double)maxHeight / source.PixelHeight);
            if (scale >= 1.0) return source;
            var scaledWidth = (int)(source.PixelWidth * scale);
            var scaledHeight = (int)(source.PixelHeight * scale);
            var group = new System.Windows.Media.TransformGroup();
            group.Children.Add(new System.Windows.Media.ScaleTransform(scale, scale));
            var transformed = new TransformedBitmap(source, group);
            transformed.Freeze();
            return (BitmapSource)transformed;
        });
    }
}
