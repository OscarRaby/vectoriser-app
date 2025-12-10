using System;
using System.IO;
using System.Windows;
using System.Windows.Threading;

namespace OrganicVectoriser;

public partial class App : Application
{
    private static readonly string LogFile = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "OrganicVectoriser",
        "error.log"
    );

    public App()
    {
        try
        {
            // Ensure log directory exists
            var dir = Path.GetDirectoryName(LogFile);
            if (!string.IsNullOrEmpty(dir))
            {
                Directory.CreateDirectory(dir);
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Failed to create log directory: {ex.Message}", "Initialization Error");
        }

        // Global exception handler for unhandled exceptions (background threads)
        AppDomain.CurrentDomain.UnhandledException += (s, e) =>
        {
            var ex = e.ExceptionObject as Exception;
            LogException("AppDomain UnhandledException", ex);
            
            Dispatcher.Invoke(() =>
            {
                MessageBox.Show(
                    $"Unhandled exception:\n\n{ex?.Message}\n\nCheck error.log for details.",
                    "Fatal Error",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error
                );
            });
        };

        // Global exception handler for dispatcher exceptions
        DispatcherUnhandledException += (s, e) =>
        {
            LogException("DispatcherUnhandledException", e.Exception);
            
            MessageBox.Show(
                $"Dispatcher error:\n\n{e.Exception?.Message}\n\nCheck error.log for details.",
                "Error",
                MessageBoxButton.OK,
                MessageBoxImage.Error
            );
            e.Handled = true;
        };

        // Handle task scheduler exceptions
        TaskScheduler.UnobservedTaskException += (s, e) =>
        {
            LogException("UnobservedTaskException", e.Exception);
            e.SetObserved();
        };
    }

    private static void LogException(string context, Exception ex)
    {
        try
        {
            var message = $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}] {context}\n" +
                         $"Message: {ex?.Message}\n" +
                         $"Type: {ex?.GetType().FullName}\n" +
                         $"StackTrace:\n{ex?.StackTrace}\n";
            
            if (ex?.InnerException != null)
            {
                message += $"\nInner Exception:\n" +
                          $"Message: {ex.InnerException.Message}\n" +
                          $"Type: {ex.InnerException.GetType().FullName}\n" +
                          $"StackTrace:\n{ex.InnerException.StackTrace}\n";
            }

            message += "\n" + new string('=', 80) + "\n\n";

            File.AppendAllText(LogFile, message);
        }
        catch
        {
            // Silently fail if we can't write to log
        }
    }
}
