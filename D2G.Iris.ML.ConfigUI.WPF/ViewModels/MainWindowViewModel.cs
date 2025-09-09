using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using System.Windows.Input;
using D2G.Iris.ML.ConfigUI.WPF.Commands;
using D2G.Iris.ML.ConfigUI.WPF.Services;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Configuration;
using D2G.Iris.ML.Data;

namespace D2G.Iris.ML.ConfigUI.WPF.ViewModels
{
    public class MainWindowViewModel : BaseViewModel
    {
        private readonly IConfigurationService _configService;
        private readonly IDialogService _dialogService;
        private ModelConfig? _currentConfig;
        private string? _currentFilePath;
        private string _windowTitle = "Iris ML Config";
        private bool _isTraining;
        private int _selectedTabIndex;

        public MainWindowViewModel(
            IConfigurationService configService,
            IDialogService dialogService)
        {
            _configService = configService;
            _dialogService = dialogService;

            InitializeViewModels();
            InitializeCommands();
            LoadExistingConfigOnStartup();
        }

        #region Properties

        public string WindowTitle
        {
            get => _windowTitle;
            set => SetProperty(ref _windowTitle, value);
        }

        public bool IsTraining
        {
            get => _isTraining;
            set => SetProperty(ref _isTraining, value);
        }

        public int SelectedTabIndex
        {
            get => _selectedTabIndex;
            set => SetProperty(ref _selectedTabIndex, value);
        }

        public GeneralSettingsViewModel GeneralSettings { get; private set; } = null!;
        public DatabaseSettingsViewModel DatabaseSettings { get; private set; } = null!;
        public InputFieldsViewModel InputFields { get; private set; } = null!;
        public ExploratoryDataAnalysisViewModel ExploratoryDataAnalysis { get; private set; } = null!;
        public TrainingParametersViewModel TrainingParameters { get; private set; } = null!;
        public DataProcessingPipelineViewModel DataProcessingPipeline { get; private set; } = null!;
        public TrainingLogsViewModel TrainingLogs { get; private set; } = null!;

        #endregion

        #region Commands

        public ICommand NewConfigCommand { get; private set; } = null!;
        public ICommand OpenConfigCommand { get; private set; } = null!;
        public ICommand SaveConfigCommand { get; private set; } = null!;
        public ICommand ExitCommand { get; private set; } = null!;
        public ICommand LaunchTrainingCommand { get; private set; } = null!;

        #endregion

        private void InitializeViewModels()
        {
            GeneralSettings = new GeneralSettingsViewModel();
            DatabaseSettings = new DatabaseSettingsViewModel(_dialogService);
            InputFields = new InputFieldsViewModel(_dialogService);
            ExploratoryDataAnalysis = new ExploratoryDataAnalysisViewModel(_dialogService);
            TrainingParameters = new TrainingParametersViewModel(_dialogService);
            DataProcessingPipeline = new DataProcessingPipelineViewModel();
            TrainingLogs = new TrainingLogsViewModel();

            TrainingParameters.ModelTypeChanged += OnModelTypeChanged;

            InputFields.SetDependencies(() => DatabaseSettings.GetConfiguration(), () => TrainingParameters.TargetField);
            ExploratoryDataAnalysis.SetDependencies(() => DatabaseSettings.GetConfiguration(), () => InputFields.GetConfiguration(), () => TrainingParameters.TargetField);
        }

        private void InitializeCommands()
        {
            NewConfigCommand = new RelayCommand(CreateNewConfiguration);
            OpenConfigCommand = new RelayCommand(OpenConfiguration);
            SaveConfigCommand = new RelayCommand(SaveConfiguration);
            ExitCommand = new RelayCommand(_ => System.Windows.Application.Current.Shutdown());
            LaunchTrainingCommand = new AsyncRelayCommand(LaunchTraining, () => !IsTraining);
        }

        private void OnModelTypeChanged(ModelType newModelType)
        {
        }

        private void LoadExistingConfigOnStartup()
        {
            try
            {
                string appConfigPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "modelconfig.json");

                if (File.Exists(appConfigPath))
                {
                    try
                    {
                        _currentConfig = _configService.LoadConfiguration(appConfigPath);
                        _currentFilePath = appConfigPath;
                        UpdateFormTitle();
                        UpdateUIFromConfig();

                        TrainingLogs.LogMessage("Successfully loaded configuration from: " + appConfigPath, "Success");
                        return;
                    }
                    catch (Exception loadEx)
                    {
                        TrainingLogs.LogMessage($"Error loading configuration: {loadEx.Message}", "Error");
                    }
                }

                CreateNewConfiguration();
            }
            catch (Exception ex)
            {
                CreateNewConfiguration();
                TrainingLogs.LogMessage($"Unexpected error: {ex.Message}", "Error");
                TrainingLogs.LogMessage("Created a new configuration.", "Warning");
            }
        }

        private void CreateNewConfiguration()
        {
            _currentConfig = new ModelConfig
            {
                Author = Environment.UserName,
                Description = "New Model Configuration",
                ModelType = ModelType.BinaryClassification,
                TargetField = "Label",
                Database = new DatabaseConfig
                {
                    Server = "localhost",
                    Database = "IrisData",
                    TableName = "DataTable",
                    OutputTableName = "",
                    WhereClause = ""
                },
                TrainingParameters = new TrainingParameters
                {
                    Algorithm = "fasttree",
                    TestFraction = 0.2,
                    AlgorithmParameters = new Dictionary<string, object>
                    {
                        { "NumberOfLeaves", 20 }
                    }
                },
                InputFields = new List<InputField>(),
                FeatureEngineering = new FeatureEngineeringConfig
                {
                    Method = FeatureSelectionMethod.None,
                    ExecutionOrder = 2,
                    NumberOfComponents = 3,
                    MaxFeatures = 10,
                    MulticollinearityThreshold = 0.7
                },
                DataBalancing = new DataBalancingConfig
                {
                    Method = DataBalanceMethod.None,
                    ExecutionOrder = 1,
                    KNeighbors = 5,
                    UndersamplingRatio = 0.9f,
                    MinorityToMajorityRatio = 0.1f
                },
                AutoML = new AutoMLConfig
                {
                    Enabled = false,
                    MaxExperimentTimeInSeconds = 30,
                    MaxModels = 10,
                    OptimizingMetric = "Accuracy"
                }
            };

            _currentFilePath = null;
            UpdateFormTitle();
            UpdateUIFromConfig();

            TrainingLogs.LogMessage("Created new configuration", "Info");
        }

        private void OpenConfiguration()
        {
            var filePath = _dialogService.ShowOpenFileDialog("JSON files (*.json)|*.json|All files (*.*)|*.*");
            if (filePath != null)
            {
                try
                {
                    _currentConfig = _configService.LoadConfiguration(filePath);
                    _currentFilePath = filePath;
                    UpdateFormTitle();
                    UpdateUIFromConfig();

                    TrainingLogs.LogMessage($"Loaded configuration from: {_currentFilePath}", "Success");
                    _dialogService.ShowInfoDialog("Configuration loaded successfully.", "Success");
                }
                catch (Exception ex)
                {
                    TrainingLogs.LogMessage($"Error loading configuration: {ex.Message}", "Error");
                    _dialogService.ShowErrorDialog($"Error loading configuration: {ex.Message}", "Error");
                }
            }
        }

        private void SaveConfiguration()
        {
            UpdateConfigFromUI();

            if (string.IsNullOrEmpty(_currentFilePath))
            {
                var filePath = _dialogService.ShowSaveFileDialog(
                    "JSON files (*.json)|*.json|All files (*.*)|*.*",
                    "json",
                    "modelconfig.json");

                if (filePath == null) return;

                _currentFilePath = filePath;
            }

            try
            {
                _configService.SaveConfiguration(_currentConfig!, _currentFilePath);
                UpdateFormTitle();
                TrainingLogs.LogMessage($"Configuration saved to: {_currentFilePath}", "Success");
                _dialogService.ShowInfoDialog("Configuration saved successfully.", "Success");
            }
            catch (Exception ex)
            {
                TrainingLogs.LogMessage($"Error saving configuration: {ex.Message}", "Error");
                _dialogService.ShowErrorDialog($"Error saving configuration: {ex.Message}", "Error");
            }
        }

        private async Task LaunchTraining()
        {
            try
            {
                if (_currentConfig == null)
                {
                    _dialogService.ShowErrorDialog("Please create a configuration first.", "No Configuration");
                    return;
                }

                UpdateConfigFromUI();

                if (!_configService.ValidateConfiguration(_currentConfig))
                {
                    _dialogService.ShowErrorDialog("Configuration validation failed. Please check all required fields.", "Validation Error");
                    return;
                }

                string confirmationMessage = _currentConfig.AutoML?.Enabled == true
                    ? $"Are you sure you want to start AutoML training? This will run for up to {_currentConfig.AutoML.MaxExperimentTimeInSeconds} seconds."
                    : "Are you sure you want to start the training process?";

                if (string.IsNullOrEmpty(_currentFilePath))
                {
                    if (_dialogService.ShowConfirmationDialog("Configuration needs to be saved before training. Save now?", "Save Required"))
                    {
                        SaveConfiguration();
                    }
                    else
                    {
                        return;
                    }
                }
                else
                {
                    _configService.SaveConfiguration(_currentConfig, _currentFilePath);
                }

                if (!_dialogService.ShowConfirmationDialog(confirmationMessage, "Confirm Training"))
                {
                    return;
                }

                SelectedTabIndex = 6;

                TrainingLogs.ClearLogs();
                IsTraining = true;

                await RunTrainingProcess();

                _dialogService.ShowInfoDialog("Training process completed! Check the Training Logs tab for details.", "Training Complete");
            }
            catch (Exception ex)
            {
                TrainingLogs.LogMessage($"ERROR: {ex.Message}", "Error");
                if (ex.InnerException != null)
                {
                    TrainingLogs.LogMessage($"Inner exception: {ex.InnerException.Message}", "Error");
                }
                TrainingLogs.LogMessage($"Stack trace: {ex.StackTrace}", "Error");

                _dialogService.ShowErrorDialog($"Error during training: {ex.Message}", "Training Error");
            }
            finally
            {
                IsTraining = false;
            }
        }

        private async Task RunTrainingProcess()
        {
            await Task.Run(() =>
            {
                try
                {
                    string tempConfigPath = Path.Combine(Path.GetTempPath(), "modelconfig.json");
                    var serializableConfig = new Dictionary<string, ModelConfig>
                    {
                        { "modelConfig", _currentConfig! }
                    };

                    var options = new JsonSerializerOptions
                    {
                        WriteIndented = true,
                        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                        Converters = { new JsonStringEnumConverter() }
                    };

                    string jsonConfig = JsonSerializer.Serialize(serializableConfig, options);
                    File.WriteAllText(tempConfigPath, jsonConfig);

                    var configManager = new ConfigManager();
                    var config = configManager.LoadConfiguration(tempConfigPath);

                    var sqlHandler = new SqlHandler(config.Database.TableName);
                    sqlHandler.Connect(config.Database);

                    var enabledFields = config.InputFields
                        .Where(f => f.IsEnabled)
                        .Select(f => f.Name)
                        .ToArray();

                    var mlContext = new Microsoft.ML.MLContext(seed: 42);
                    Microsoft.ML.IDataView rawData;

                    // Check if we have cleaned data from EDA outlier removal
                    if (ExploratoryDataAnalysis.HasDataBeenCleaned())
                    {
                        rawData = ExploratoryDataAnalysis.GetCleanedDataAsIDataView(
                            mlContext, 
                            enabledFields, 
                            config.TargetField, 
                            config.ModelType);
                        
                        if (rawData != null)
                        {
                            var cleanedRowCount = ExploratoryDataAnalysis.GetCleanedDataForTraining()?.Rows.Count ?? 0;
                            Console.WriteLine("=============== Loading Data ===============");
                            Console.WriteLine($"Using cleaned dataset from EDA outlier removal.");
                            Console.WriteLine($">> Loaded {cleanedRowCount} rows of cleaned data.");
                        }
                        else
                        {
                            // Fallback to original data loading
                            var dataLoader = new DatabaseDataLoader();
                            rawData = dataLoader.LoadDataFromSql(
                                sqlHandler.GetConnectionString(),
                                config.Database.TableName,
                                enabledFields,
                                config.ModelType,
                                config.TargetField,
                                config.Database.WhereClause);
                        }
                    }
                    else
                    {
                        // Original data loading
                        var dataLoader = new DatabaseDataLoader();
                        rawData = dataLoader.LoadDataFromSql(
                            sqlHandler.GetConnectionString(),
                            config.Database.TableName,
                            enabledFields,
                            config.ModelType,
                            config.TargetField,
                            config.Database.WhereClause);
                    }

                    var dataProcessor = new DataProcessor(sqlHandler);
                    var processedData = dataProcessor.ProcessData(
                        mlContext,
                        rawData,
                        enabledFields,
                        config).GetAwaiter().GetResult();

                    var modelTrainerFactory = new Training.ModelTrainerFactory(mlContext);
                    var modelTrainer = modelTrainerFactory.CreateTrainer(config.ModelType);

                    modelTrainer.TrainModel(
                        mlContext,
                        processedData.Data,
                        processedData.FeatureNames,
                        config,
                        processedData).GetAwaiter().GetResult();

                    try { File.Delete(tempConfigPath); } catch { }
                }
                catch (Exception ex)
                {
                    TrainingLogs.LogMessage($"Error in training process: {ex.Message}", "Error");
                    throw;
                }
            });
        }

        private void UpdateFormTitle()
        {
            string fileName = Path.GetFileName(_currentFilePath) ?? "Untitled";
            WindowTitle = $"Iris ML Config - {fileName}";
        }

        private void UpdateUIFromConfig()
        {
            if (_currentConfig == null) return;

            GeneralSettings.SetConfiguration(_currentConfig);
            DatabaseSettings.SetConfiguration(_currentConfig.Database);
            InputFields.SetConfiguration(_currentConfig.InputFields);

            TrainingParameters.SetConfiguration(
                _currentConfig.TrainingParameters,
                _currentConfig.AutoML,
                _currentConfig.ModelType,
                _currentConfig.TargetField);
            DataProcessingPipeline.LoadFromConfig(_currentConfig);
        }

        private void UpdateConfigFromUI()
        {
            if (_currentConfig == null) return;

            GeneralSettings.UpdateConfiguration(_currentConfig);
            _currentConfig.Database = DatabaseSettings.GetConfiguration();
            _currentConfig.InputFields = InputFields.GetConfiguration();

            var (trainingParams, autoMLConfig, modelType, targetField) = TrainingParameters.GetConfiguration();
            _currentConfig.TrainingParameters = trainingParams;
            _currentConfig.AutoML = autoMLConfig;
            _currentConfig.ModelType = modelType;
            _currentConfig.TargetField = targetField;

            DataProcessingPipeline.SaveToConfig(_currentConfig);
        }
    }
}