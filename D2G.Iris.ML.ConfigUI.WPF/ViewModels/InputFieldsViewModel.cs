using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using D2G.Iris.ML.ConfigUI.WPF.Commands;
using D2G.Iris.ML.ConfigUI.WPF.Models;
using D2G.Iris.ML.ConfigUI.WPF.Services;
using D2G.Iris.ML.ConfigUI.WPF.Dialogs;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.ConfigUI.WPF.ViewModels
{
    public class InputFieldsViewModel : BaseViewModel
    {
        private readonly IDialogService _dialogService;
        private readonly IDatabaseSchemaLoader _schemaLoader;
        private InputFieldItem? _selectedField;
        private bool _isLoadingFromDatabase;
        private Func<DatabaseConfig?>? _getDatabaseConfig;
        private Func<string>? _getTargetField;

        public InputFieldsViewModel(IDialogService dialogService)
        {
            _dialogService = dialogService;
            _schemaLoader = new DatabaseSchemaLoader();
            InputFields = new ObservableCollection<InputFieldItem>();
            InputFields.CollectionChanged += (s, e) => 
            {
                ((RelayCommand)SelectAllCommand).RaiseCanExecuteChanged();
                ((RelayCommand)DeselectAllCommand).RaiseCanExecuteChanged();
            };
            InitializeCommands();
        }

        #region Properties

        public ObservableCollection<InputFieldItem> InputFields { get; }

        public InputFieldItem? SelectedField
        {
            get => _selectedField;
            set => SetProperty(ref _selectedField, value);
        }

        public bool IsLoadingFromDatabase
        {
            get => _isLoadingFromDatabase;
            set => SetProperty(ref _isLoadingFromDatabase, value);
        }

        #endregion

        #region Commands

        public ICommand AddFieldCommand { get; private set; } = null!;
        public ICommand EditFieldCommand { get; private set; } = null!;
        public ICommand RemoveFieldCommand { get; private set; } = null!;
        public ICommand LoadFromDatabaseCommand { get; private set; } = null!;
        public ICommand SelectAllCommand { get; private set; } = null!;
        public ICommand DeselectAllCommand { get; private set; } = null!;

        #endregion

        private void InitializeCommands()
        {
            AddFieldCommand = new RelayCommand(AddField);
            EditFieldCommand = new RelayCommand(EditField, () => SelectedField != null);
            RemoveFieldCommand = new RelayCommand(RemoveField, () => SelectedField != null);
            LoadFromDatabaseCommand = new AsyncRelayCommand(LoadFromDatabase, () => !IsLoadingFromDatabase);
            SelectAllCommand = new RelayCommand(SelectAll, () => InputFields.Any());
            DeselectAllCommand = new RelayCommand(DeselectAll, () => InputFields.Any());
        }

        public void SetDependencies(Func<DatabaseConfig?> getDatabaseConfig, Func<string> getTargetField)
        {
            _getDatabaseConfig = getDatabaseConfig;
            _getTargetField = getTargetField;
        }

        private void AddField()
        {
            var dialogViewModel = new InputFieldDialogViewModel
            {
                Title = "Add Input Field",
                FieldName = "",
                IsEnabled = true
            };

            var dialog = new InputFieldDialog
            {
                DataContext = dialogViewModel,
                Owner = System.Windows.Application.Current.MainWindow
            };

            if (dialog.ShowDialog() == true)
            {
                var existingField = InputFields.FirstOrDefault(f =>
                    string.Equals(f.Name, dialogViewModel.FieldName, StringComparison.OrdinalIgnoreCase));

                if (existingField != null)
                {
                    _dialogService.ShowErrorDialog($"A field with the name '{dialogViewModel.FieldName}' already exists.", "Duplicate Field");
                    return;
                }

                InputFields.Add(new InputFieldItem
                {
                    Name = dialogViewModel.FieldName,
                    IsEnabled = dialogViewModel.IsEnabled
                });
            }
        }

        private void EditField()
        {
            if (SelectedField == null) return;

            var dialogViewModel = new InputFieldDialogViewModel
            {
                Title = "Edit Input Field",
                FieldName = SelectedField.Name,
                IsEnabled = SelectedField.IsEnabled
            };

            var dialog = new InputFieldDialog
            {
                DataContext = dialogViewModel,
                Owner = System.Windows.Application.Current.MainWindow
            };

            if (dialog.ShowDialog() == true)
            {
                var existingField = InputFields.FirstOrDefault(f =>
                    f != SelectedField &&
                    string.Equals(f.Name, dialogViewModel.FieldName, StringComparison.OrdinalIgnoreCase));

                if (existingField != null)
                {
                    _dialogService.ShowErrorDialog($"A field with the name '{dialogViewModel.FieldName}' already exists.", "Duplicate Field");
                    return;
                }

                SelectedField.Name = dialogViewModel.FieldName;
                SelectedField.IsEnabled = dialogViewModel.IsEnabled;
            }
        }

        private void RemoveField()
        {
            if (SelectedField == null) return;

            if (_dialogService.ShowConfirmationDialog(
                $"Are you sure you want to remove the field '{SelectedField.Name}'?",
                "Confirm Removal"))
            {
                InputFields.Remove(SelectedField);
            }
        }

        private void SelectAll()
        {
            foreach (var field in InputFields)
            {
                field.IsEnabled = true;
            }
        }

        private void DeselectAll()
        {
            foreach (var field in InputFields)
            {
                field.IsEnabled = false;
            }
        }

        private async Task LoadFromDatabase()
        {
            try
            {
                var dbConfig = _getDatabaseConfig?.Invoke();
                var targetField = _getTargetField?.Invoke();

                if (dbConfig == null)
                {
                    _dialogService.ShowInfoDialog("Please configure database settings first.", "Database Configuration Required");
                    return;
                }

                if (string.IsNullOrWhiteSpace(dbConfig.TableName))
                {
                    _dialogService.ShowInfoDialog("Please specify a table name in database settings.", "Table Name Required");
                    return;
                }

                IsLoadingFromDatabase = true;

                if (!_schemaLoader.TestConnection(dbConfig))
                {
                    _dialogService.ShowErrorDialog("Cannot connect to database. Please check your database settings.", "Connection Failed");
                    return;
                }

                var columnNames = await Task.Run(() => _schemaLoader.LoadTableColumns(dbConfig));

                var inputFields = columnNames
                    .Where(name => !string.Equals(name, targetField, StringComparison.OrdinalIgnoreCase))
                    .Select(name => new InputFieldItem
                    {
                        Name = name,
                        IsEnabled = true
                    })
                    .OrderBy(f => f.Name)
                    .ToList();

                if (InputFields.Count > 0)
                {
                    var result = _dialogService.ShowConfirmationDialog(
                        $"This will replace your current {InputFields.Count} field(s) with {inputFields.Count} fields from the database table '{dbConfig.TableName}'.\n\n" +
                        "Do you want to continue?",
                        "Replace Current Fields?");

                    if (!result)
                        return;
                }

                InputFields.Clear();
                foreach (var field in inputFields)
                {
                    InputFields.Add(field);
                }

                _dialogService.ShowInfoDialog(
                    $"Successfully loaded {inputFields.Count} fields from database table '{dbConfig.TableName}'.",
                    "Fields Loaded Successfully");
            }
            catch (Exception ex)
            {
                _dialogService.ShowErrorDialog($"Error loading fields from database:\n\n{ex.Message}", "Database Error");
            }
            finally
            {
                IsLoadingFromDatabase = false;
            }
        }

        public void SetConfiguration(List<InputField>? inputFields)
        {
            InputFields.Clear();

            if (inputFields != null)
            {
                foreach (var field in inputFields)
                {
                    InputFields.Add(new InputFieldItem
                    {
                        Name = field.Name,
                        IsEnabled = field.IsEnabled
                    });
                }
            }
        }

        public List<InputField> GetConfiguration()
        {
            return InputFields.Select(f => new InputField
            {
                Name = f.Name,
                IsEnabled = f.IsEnabled
            }).ToList();
        }
    }

    public class InputFieldDialogViewModel : BaseViewModel
    {
        private string _title = "Input Field";
        private string _fieldName = "";
        private bool _isEnabled = true;

        public string Title
        {
            get => _title;
            set => SetProperty(ref _title, value);
        }

        public string FieldName
        {
            get => _fieldName;
            set => SetProperty(ref _fieldName, value);
        }

        public bool IsEnabled
        {
            get => _isEnabled;
            set => SetProperty(ref _isEnabled, value);
        }
    }
}