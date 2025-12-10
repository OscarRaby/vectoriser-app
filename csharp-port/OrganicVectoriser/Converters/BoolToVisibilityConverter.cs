using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace OrganicVectoriser.Converters;

public sealed class BoolToVisibilityConverter : IValueConverter
{
    public bool Invert { get; set; }
    public Visibility FalseVisibility { get; set; } = Visibility.Collapsed;

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        var flag = value is bool b && b;
        if (parameter is string s && bool.TryParse(s, out var parsed))
        {
            flag = parsed ? !flag : flag;
        }
        if (Invert)
        {
            flag = !flag;
        }
        return flag ? Visibility.Visible : FalseVisibility;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        return value is Visibility v && v == Visibility.Visible;
    }
}
