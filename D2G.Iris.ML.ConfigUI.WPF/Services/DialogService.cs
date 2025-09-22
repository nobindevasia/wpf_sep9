using System;
using System.Windows;
using Microsoft.Win32;
using D2G.Iris.ML.ConfigUI.WPF.Dialogs;

namespace D2G.Iris.ML.ConfigUI.WPF.Services
{
    public class DialogService : IDialogService
    {
        public bool ShowConfirmationDialog(string message, string title)
        {
            var result = MessageBox.Show(message, title, MessageBoxButton.YesNo, MessageBoxImage.Question);
            return result == MessageBoxResult.Yes;
        }

        public void ShowErrorDialog(string message, string title)
        {
            MessageBox.Show(message, title, MessageBoxButton.OK, MessageBoxImage.Error);
        }
       
        public void ShowInfoDialog(string message, string title)
        {
            MessageBox.Show(message, title, MessageBoxButton.OK, MessageBoxImage.Information);
        }

        public string? ShowSaveFileDialog(string filter, string defaultExtension, string defaultFileName)
        {
            var dialog = new SaveFileDialog
            {
                Filter = filter,
                DefaultExt = defaultExtension,
                FileName = defaultFileName
            };

            return dialog.ShowDialog() == true ? dialog.FileName : null;
        }

        public string? ShowOpenFileDialog(string filter)
        {
            var dialog = new OpenFileDialog
            {
                Filter = filter
            };

            return dialog.ShowDialog() == true ? dialog.FileName : null;
        }

        public T? ShowDialog<T>(object viewModel) where T : class
        {
            Window? dialog = null;

            try
            {
                if (typeof(T) == typeof(InputFieldDialog))
                {
                    dialog = new InputFieldDialog { DataContext = viewModel };
                }
                else if (typeof(T) == typeof(ParameterDialog))
                {
                    dialog = new ParameterDialog { DataContext = viewModel };
                }

                if (dialog != null)
                {
                    dialog.Owner = Application.Current.MainWindow;
                    var result = dialog.ShowDialog();
                    return result == true ? dialog as T : null;
                }
            }
            catch (Exception ex)
            {
                ShowErrorDialog($"Error showing dialog: {ex.Message}", "Dialog Error");
            }

            return null;
        }

        public bool? ShowInputFieldDialog(object viewModel)
        {
            try
            {
                var dialog = new InputFieldDialog
                {
                    DataContext = viewModel,
                    Owner = Application.Current.MainWindow
                };
                return dialog.ShowDialog();
            }
            catch (Exception ex)
            {
                ShowErrorDialog($"Error showing input field dialog: {ex.Message}", "Dialog Error");
                return false;
            }
        }

        public bool? ShowParameterDialog(object viewModel)
        {
            try
            {
                var dialog = new ParameterDialog
                {
                    DataContext = viewModel,
                    Owner = Application.Current.MainWindow
                };
                return dialog.ShowDialog();
            }
            catch (Exception ex)
            {
                ShowErrorDialog($"Error showing parameter dialog: {ex.Message}", "Dialog Error");
                return false;
            }
        }
    }
}