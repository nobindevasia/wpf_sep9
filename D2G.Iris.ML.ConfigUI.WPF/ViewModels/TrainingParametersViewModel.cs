using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows.Input;
using D2G.Iris.ML.ConfigUI.WPF.Commands;
using D2G.Iris.ML.ConfigUI.WPF.Models;
using D2G.Iris.ML.ConfigUI.WPF.Services;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;

namespace D2G.Iris.ML.ConfigUI.WPF.ViewModels
{
    public class TrainingParametersViewModel : BaseViewModel
    {
        private readonly IDialogService _dialogService;
        private string _selectedAlgorithm = "fasttree";
        private decimal _testFraction = 0.2m;
        private ModelType _currentModelType = ModelType.BinaryClassification;
        private Models.ParameterItem? _selectedParameter;

    
        private bool _useAutoML = false;
        private int _maxExperimentTimeInSeconds = 30;
        private int _maxModels = 10;
        private string _optimizingMetric = "Accuracy";

     
        private ModelType _modelType = ModelType.BinaryClassification;
        private string _targetField = "Label";

        public event Action<ModelType>? ModelTypeChanged;

        public TrainingParametersViewModel(IDialogService dialogService)
        {
            _dialogService = dialogService;
            Parameters = new ObservableCollection<Models.ParameterItem>();
            InitializeCommands();
            UpdateAvailableAlgorithms();
            UpdateAvailableMetrics();
        }

        #region Properties

        public bool UseAutoML
        {
            get => _useAutoML;
            set
            {
                if (SetProperty(ref _useAutoML, value))
                {
                    OnPropertyChanged(nameof(UseTraditionalTraining));
                    OnPropertyChanged(nameof(TrainingModeDescription));
                }
            }
        }

        public bool UseTraditionalTraining
        {
            get => !_useAutoML;
            set => UseAutoML = !value;
        }

        public string TrainingModeDescription
        {
            get => UseAutoML
                ? "AutoML will automatically try multiple algorithms and find the best performing model. Data splitting is handled automatically using cross-validation."
                : "Traditional training uses the specified algorithm with configured parameters. You can specify the test fraction for evaluation.";
        }

        public string SelectedAlgorithm
        {
            get => _selectedAlgorithm;
            set
            {
                if (SetProperty(ref _selectedAlgorithm, value))
                {
                    OnAlgorithmChanged();
                }
            }
        }

        public decimal TestFraction
        {
            get => _testFraction;
            set => SetProperty(ref _testFraction, value);
        }

        public ObservableCollection<string> AvailableAlgorithms { get; } = new();

        public ObservableCollection<Models.ParameterItem> Parameters { get; }

        public Models.ParameterItem? SelectedParameter
        {
            get => _selectedParameter;
            set => SetProperty(ref _selectedParameter, value);
        }

        public int MaxExperimentTimeInSeconds
        {
            get => _maxExperimentTimeInSeconds;
            set => SetProperty(ref _maxExperimentTimeInSeconds, Math.Max(1, Math.Min(3600, value)));
        }

        public int MaxModels
        {
            get => _maxModels;
            set => SetProperty(ref _maxModels, Math.Max(1, Math.Min(100, value)));
        }

        public string OptimizingMetric
        {
            get => _optimizingMetric;
            set => SetProperty(ref _optimizingMetric, value);
        }

        public ObservableCollection<string> AvailableMetrics { get; } = new();

        public ModelType ModelType
        {
            get => _modelType;
            set
            {
                if (SetProperty(ref _modelType, value))
                {
                    _currentModelType = value;
                    ModelTypeChanged?.Invoke(value);
                    UpdateAvailableAlgorithms();
                    UpdateAvailableMetrics();
                    Parameters.Clear();
                }
            }
        }

        public string TargetField
        {
            get => _targetField;
            set => SetProperty(ref _targetField, value);
        }

        public IEnumerable<ModelType> AvailableModelTypes => Enum.GetValues<ModelType>();

        #endregion

        #region Commands

        public ICommand AddParameterCommand { get; private set; } = null!;
        public ICommand RemoveParameterCommand { get; private set; } = null!;

        #endregion

        private void InitializeCommands()
        {
            AddParameterCommand = new RelayCommand(_ => AddParameter());
            RemoveParameterCommand = new RelayCommand(_ => RemoveParameter(), _ => SelectedParameter != null);
        }

        public void SetModelType(ModelType modelType)
        {
            if (_currentModelType != modelType)
            {
                _currentModelType = modelType;
                _modelType = modelType; 
                UpdateAvailableAlgorithms();
                Parameters.Clear();
                UpdateAvailableMetrics();
                OnPropertyChanged(nameof(ModelType));
            }
        }

        private void UpdateAvailableAlgorithms()
        {
            AvailableAlgorithms.Clear();
            var algorithms = AlgorithmRegistry.GetAlgorithmsForModelType(_currentModelType);

            foreach (var algorithm in algorithms)
            {
                AvailableAlgorithms.Add(algorithm);
            }

            if (AvailableAlgorithms.Count > 0 && !AvailableAlgorithms.Contains(SelectedAlgorithm))
            {
                SelectedAlgorithm = AvailableAlgorithms[0];
            }
        }

        private void UpdateAvailableMetrics()
        {
            AvailableMetrics.Clear();

            switch (_currentModelType)
            {
                case ModelType.BinaryClassification:
                    foreach (var metric in new[] { "Accuracy", "AUC", "F1Score" })
                        AvailableMetrics.Add(metric);
                    OptimizingMetric = "Accuracy";
                    break;
                case ModelType.MultiClassClassification:
                    foreach (var metric in new[] { "MicroAccuracy", "MacroAccuracy" })
                        AvailableMetrics.Add(metric);
                    OptimizingMetric = "MicroAccuracy";
                    break;
                case ModelType.Regression:
                    foreach (var metric in new[] { "RSquared", "MeanAbsoluteError", "RootMeanSquaredError" })
                        AvailableMetrics.Add(metric);
                    OptimizingMetric = "RSquared";
                    break;
            }
        }

        private void OnAlgorithmChanged()
        {
            Parameters.Clear();
            LoadAvailableParameters(SelectedAlgorithm);
        }

        private void LoadAvailableParameters(string algorithmName)
        {
            try
            {
                Type? optionsType = AlgorithmRegistry.GetOptionsType(algorithmName, _currentModelType);
                if (optionsType == null) return;

                Parameters.Clear();
                OnPropertyChanged(nameof(AlgorithmTooltip));
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error loading parameters for {algorithmName}: {ex.Message}");
            }
        }

        public string AlgorithmTooltip
        {
            get
            {
                try
                {
                    var optionsType = AlgorithmRegistry.GetOptionsType(_selectedAlgorithm, _currentModelType);
                    return optionsType != null
                        ? ParameterHelper.CreateParameterTooltip(optionsType)
                        : "No parameter information available.";
                }
                catch
                {
                    return "Error loading parameter information.";
                }
            }
        }

        private void AddParameter()
        {
            var dialogViewModel = new ParameterDialogViewModel(_selectedAlgorithm, _currentModelType);
            var result = _dialogService.ShowParameterDialog(dialogViewModel);

            if (result == true)
            {
                var existingParam = Parameters.FirstOrDefault(p =>
                    string.Equals(p.Name, dialogViewModel.ParameterName, StringComparison.OrdinalIgnoreCase));

                if (existingParam != null)
                {
                    if (_dialogService.ShowConfirmationDialog(
                        $"Parameter '{dialogViewModel.ParameterName}' already exists. Do you want to update it?",
                        "Duplicate Parameter"))
                    {
                        existingParam.Value = dialogViewModel.ParameterValue;
                    }
                    return;
                }

                var parameterItem = new Models.ParameterItem
                {
                    Name = dialogViewModel.ParameterName,
                    Value = dialogViewModel.ParameterValue,
                    DisplayText = $"{dialogViewModel.ParameterName} ({dialogViewModel.ParameterValue?.GetType().Name ?? "object"})"
                };

                if (dialogViewModel.ParameterValue != null)
                {
                    parameterItem.ExpectedType = dialogViewModel.ParameterValue.GetType();
                }

                Parameters.Add(parameterItem);
            }
        }

        private void RemoveParameter()
        {
            if (SelectedParameter == null) return;

            if (_dialogService.ShowConfirmationDialog(
                $"Are you sure you want to remove the parameter '{SelectedParameter.Name}'?",
                "Confirm Removal"))
            {
                Parameters.Remove(SelectedParameter);
            }
        }


        public void SetConfiguration(TrainingParameters? parameters, AutoMLConfig? autoMLConfig, ModelType modelType, string targetField)
        {
            ModelType = modelType;
            TargetField = targetField ?? "Label";

            if (autoMLConfig != null)
            {
                UseAutoML = autoMLConfig.Enabled;
                MaxExperimentTimeInSeconds = autoMLConfig.MaxExperimentTimeInSeconds;
                MaxModels = autoMLConfig.MaxModels;
                OptimizingMetric = autoMLConfig.OptimizingMetric ?? "Accuracy";
            }
            else
            {
                UseAutoML = false;
                MaxExperimentTimeInSeconds = 30;
                MaxModels = 10;
                OptimizingMetric = "Accuracy";
            }

            if (parameters == null) return;

            if (!string.IsNullOrEmpty(parameters.Algorithm))
            {
                var matchingAlgorithm = AvailableAlgorithms.FirstOrDefault(a =>
                    string.Equals(a, parameters.Algorithm, StringComparison.OrdinalIgnoreCase));

                SelectedAlgorithm = matchingAlgorithm ?? (AvailableAlgorithms.Count > 0 ? AvailableAlgorithms[0] : "fasttree");
            }

            TestFraction = (decimal)parameters.TestFraction;

            Parameters.Clear();
            if (parameters.AlgorithmParameters != null)
            {
                foreach (var param in parameters.AlgorithmParameters)
                {
                    var parameterItem = new Models.ParameterItem
                    {
                        Name = param.Key,
                        Value = param.Value,
                        DisplayText = $"{param.Key} ({param.Value?.GetType().Name ?? "object"})"
                    };

                    if (param.Value != null)
                    {
                        parameterItem.ExpectedType = param.Value.GetType();
                    }

                    Parameters.Add(parameterItem);
                }
            }
        }

        public (TrainingParameters trainingParams, AutoMLConfig autoMLConfig, ModelType modelType, string targetField) GetConfiguration()
        {
            var algorithmParameters = new Dictionary<string, object>();
            foreach (var param in Parameters)
            {
                if (!string.IsNullOrEmpty(param.Name) && param.Value != null)
                {
                    var cleanName = param.Name.Split('(')[0].Trim();
                    algorithmParameters[cleanName] = param.Value;
                }
            }

            var trainingParams = new TrainingParameters
            {
                Algorithm = SelectedAlgorithm,
                TestFraction = (double)TestFraction,
                AlgorithmParameters = algorithmParameters
            };

            var autoMLConfig = new AutoMLConfig
            {
                Enabled = UseAutoML,
                MaxExperimentTimeInSeconds = MaxExperimentTimeInSeconds,
                MaxModels = MaxModels,
                OptimizingMetric = OptimizingMetric
            };

            return (trainingParams, autoMLConfig, ModelType, TargetField);
        }
    }
}