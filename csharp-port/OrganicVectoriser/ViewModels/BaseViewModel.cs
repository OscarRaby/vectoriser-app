using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace OrganicVectoriser.ViewModels;

public abstract class BaseViewModel : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;

    protected void RaisePropertyChanged([CallerMemberName] string? propertyName = null)
    {
        if (propertyName is null) return;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    protected bool SetProperty<T>(ref T storage, T value, [CallerMemberName] string? propertyName = null)
    {
        if (Equals(storage, value)) return false;
        storage = value;
        RaisePropertyChanged(propertyName);
        return true;
    }
}
