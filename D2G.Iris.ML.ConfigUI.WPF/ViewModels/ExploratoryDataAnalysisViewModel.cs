using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Threading;
using Microsoft.Data.SqlClient;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.ConfigUI.WPF.Commands;
using D2G.Iris.ML.ConfigUI.WPF.Services;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Data;

namespace D2G.Iris.ML.ConfigUI.WPF.ViewModels
{
    public class ExploratoryDataAnalysisViewModel : BaseViewModel
    {
        private readonly IDialogService _dialogService;
        private int _numberOfRows;
        private int _numberOfColumns;
        private int _totalMissingValues;
        private double _missingValuesPercentage;
        private ObservableCollection<FeatureTypeInfo> _featureTypes;
        private ObservableCollection<ColumnMissingInfo> _columnMissingValues;
        private Func<DatabaseConfig>? _getDatabaseConfig;
        private Func<List<InputField>>? _getInputFields;
        private Func<string>? _getTargetField;
        private bool _isLoading;
        private string _loadingMessage = "Loading data...";
        private VisualisationViewModel _visualisationViewModel;
        private OutlierDetectionViewModel _outlierDetectionViewModel;

        public ExploratoryDataAnalysisViewModel(IDialogService dialogService)
        {
            _dialogService = dialogService;
            _featureTypes = new ObservableCollection<FeatureTypeInfo>();
            _columnMissingValues = new ObservableCollection<ColumnMissingInfo>();
            _visualisationViewModel = new VisualisationViewModel(dialogService);
            _outlierDetectionViewModel = new OutlierDetectionViewModel(dialogService);
            
            AnalyzeDataCommand = new RelayCommand(_ => AnalyzeData(), _ => CanAnalyzeData());
        }

        #region Properties

        public int NumberOfRows
        {
            get => _numberOfRows;
            set => SetProperty(ref _numberOfRows, value);
        }

        public int NumberOfColumns
        {
            get => _numberOfColumns;
            set => SetProperty(ref _numberOfColumns, value);
        }

        public int TotalMissingValues
        {
            get => _totalMissingValues;
            set => SetProperty(ref _totalMissingValues, value);
        }

        public double MissingValuesPercentage
        {
            get => _missingValuesPercentage;
            set => SetProperty(ref _missingValuesPercentage, value);
        }

        public ObservableCollection<FeatureTypeInfo> FeatureTypes
        {
            get => _featureTypes;
            set => SetProperty(ref _featureTypes, value);
        }

        public ObservableCollection<ColumnMissingInfo> ColumnMissingValues
        {
            get => _columnMissingValues;
            set => SetProperty(ref _columnMissingValues, value);
        }

        public bool IsLoading
        {
            get => _isLoading;
            set => SetProperty(ref _isLoading, value);
        }

        public string LoadingMessage
        {
            get => _loadingMessage;
            set => SetProperty(ref _loadingMessage, value);
        }

        public VisualisationViewModel VisualisationViewModel
        {
            get => _visualisationViewModel;
            set => SetProperty(ref _visualisationViewModel, value);
        }

        public OutlierDetectionViewModel OutlierDetectionViewModel
        {
            get => _outlierDetectionViewModel;
            set => SetProperty(ref _outlierDetectionViewModel, value);
        }

        #endregion

        #region Commands

        public ICommand AnalyzeDataCommand { get; }

        #endregion

        #region Public Methods

        public void SetDependencies(Func<DatabaseConfig> getDatabaseConfig, Func<List<InputField>> getInputFields, Func<string>? getTargetField = null)
        {
            _getDatabaseConfig = getDatabaseConfig;
            _getInputFields = getInputFields;
            _getTargetField = getTargetField;
        }

        public DataTable? GetCleanedDataForTraining()
        {
            if (_outlierDetectionViewModel.HasOutliersBeenRemoved())
            {
                return _outlierDetectionViewModel.GetCleanedDataTable();
            }
            return null;
        }

        public bool HasDataBeenCleaned()
        {
            return _outlierDetectionViewModel.HasOutliersBeenRemoved();
        }

        public IDataView? GetCleanedDataAsIDataView(MLContext mlContext, IEnumerable<string> featureColumns, string targetColumn, ModelType modelType)
        {
            if (!HasDataBeenCleaned() || _outlierDetectionViewModel.GetCleanedDataTable() == null)
                return null;

            var cleanedDataTable = _outlierDetectionViewModel.GetCleanedDataTable()!;
            
            // Validate that all required columns exist
            var featureColumnsList = featureColumns.ToList();
            var missingColumns = featureColumnsList.Where(col => !cleanedDataTable.Columns.Contains(col)).ToList();
            if (missingColumns.Any())
            {
                Console.WriteLine($"Warning: Missing feature columns: {string.Join(", ", missingColumns)}");
                featureColumnsList = featureColumnsList.Where(col => cleanedDataTable.Columns.Contains(col)).ToList();
            }

            if (!cleanedDataTable.Columns.Contains(targetColumn))
            {
                Console.WriteLine($"Error: Target column '{targetColumn}' not found in cleaned data");
                return null;
            }

            // Create the proper data view based on model type
            try
            {
                switch (modelType)
                {
                    case ModelType.BinaryClassification:
                        return CreateBinaryClassificationDataView(mlContext, cleanedDataTable, featureColumnsList.ToArray(), targetColumn);
                    
                    case ModelType.MultiClassClassification:
                        return CreateMultiClassDataView(mlContext, cleanedDataTable, featureColumnsList.ToArray(), targetColumn);
                    
                    case ModelType.Regression:
                        return CreateRegressionDataView(mlContext, cleanedDataTable, featureColumnsList.ToArray(), targetColumn);
                    
                    default:
                        throw new ArgumentException($"Unsupported model type: {modelType}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error converting DataTable to IDataView: {ex.Message}");
                return null;
            }
        }

        private IDataView CreateBinaryClassificationDataView(MLContext mlContext, DataTable dataTable, string[] featureColumns, string targetColumn)
        {
            var dataPoints = new List<BinaryClassificationDataPoint>();
            
            foreach (DataRow row in dataTable.Rows)
            {
                try
                {
                    // Extract features
                    var features = new float[featureColumns.Length];
                    bool validRow = true;
                    
                    for (int i = 0; i < featureColumns.Length; i++)
                    {
                        var value = row[featureColumns[i]];
                        if (value == null || value == DBNull.Value)
                        {
                            features[i] = 0f; // Handle missing values
                        }
                        else if (float.TryParse(value.ToString(), out float floatValue))
                        {
                            features[i] = floatValue;
                        }
                        else
                        {
                            features[i] = 0f; // Handle non-numeric values
                        }
                    }

                    // Extract label
                    var labelValue = row[targetColumn];
                    bool label = false;
                    
                    if (labelValue != null && labelValue != DBNull.Value)
                    {
                        var labelStr = labelValue.ToString().ToLower();
                        label = labelStr == "1" || labelStr == "true" || labelStr == "yes";
                    }

                    if (validRow)
                    {
                        dataPoints.Add(new BinaryClassificationDataPoint
                        {
                            Features = features,
                            Label = label
                        });
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Skipping invalid row: {ex.Message}");
                    continue;
                }
            }

            if (!dataPoints.Any())
            {
                Console.WriteLine("Error: No valid data points created");
                return null;
            }

            // Create schema definition
            var schemaDefinition = SchemaDefinition.Create(typeof(BinaryClassificationDataPoint));
            schemaDefinition[nameof(BinaryClassificationDataPoint.Features)].ColumnType = 
                new VectorDataViewType(NumberDataViewType.Single, featureColumns.Length);

            Console.WriteLine($"Created binary classification data view with {dataPoints.Count} rows and {featureColumns.Length} features");
            
            return mlContext.Data.LoadFromEnumerable(dataPoints, schemaDefinition);
        }

        private IDataView CreateMultiClassDataView(MLContext mlContext, DataTable dataTable, string[] featureColumns, string targetColumn)
        {
            var dataPoints = new List<MultiClassDataPoint>();
            
            foreach (DataRow row in dataTable.Rows)
            {
                try
                {
                    var features = new float[featureColumns.Length];
                    bool validRow = true;
                    
                    for (int i = 0; i < featureColumns.Length; i++)
                    {
                        var value = row[featureColumns[i]];
                        if (value == null || value == DBNull.Value)
                        {
                            features[i] = 0f;
                        }
                        else if (float.TryParse(value.ToString(), out float floatValue))
                        {
                            features[i] = floatValue;
                        }
                        else
                        {
                            features[i] = 0f;
                        }
                    }

                    var labelValue = row[targetColumn];
                    uint label = 0;
                    
                    if (labelValue != null && labelValue != DBNull.Value)
                    {
                        if (uint.TryParse(labelValue.ToString(), out uint parsedLabel))
                        {
                            label = parsedLabel;
                        }
                    }

                    if (validRow)
                    {
                        dataPoints.Add(new MultiClassDataPoint
                        {
                            Features = features,
                            Label = label
                        });
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Skipping invalid row: {ex.Message}");
                    continue;
                }
            }

            if (!dataPoints.Any())
            {
                Console.WriteLine("Error: No valid data points created");
                return null;
            }

            var schemaDefinition = SchemaDefinition.Create(typeof(MultiClassDataPoint));
            schemaDefinition[nameof(MultiClassDataPoint.Features)].ColumnType = 
                new VectorDataViewType(NumberDataViewType.Single, featureColumns.Length);

            Console.WriteLine($"Created multi-class data view with {dataPoints.Count} rows and {featureColumns.Length} features");
            
            return mlContext.Data.LoadFromEnumerable(dataPoints, schemaDefinition);
        }

        private IDataView CreateRegressionDataView(MLContext mlContext, DataTable dataTable, string[] featureColumns, string targetColumn)
        {
            var dataPoints = new List<RegressionDataPoint>();
            
            foreach (DataRow row in dataTable.Rows)
            {
                try
                {
                    var features = new float[featureColumns.Length];
                    bool validRow = true;
                    
                    for (int i = 0; i < featureColumns.Length; i++)
                    {
                        var value = row[featureColumns[i]];
                        if (value == null || value == DBNull.Value)
                        {
                            features[i] = 0f;
                        }
                        else if (float.TryParse(value.ToString(), out float floatValue))
                        {
                            features[i] = floatValue;
                        }
                        else
                        {
                            features[i] = 0f;
                        }
                    }

                    var labelValue = row[targetColumn];
                    float label = 0f;
                    
                    if (labelValue != null && labelValue != DBNull.Value)
                    {
                        if (float.TryParse(labelValue.ToString(), out float parsedLabel))
                        {
                            label = parsedLabel;
                        }
                    }

                    if (validRow)
                    {
                        dataPoints.Add(new RegressionDataPoint
                        {
                            Features = features,
                            Label = label
                        });
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Skipping invalid row: {ex.Message}");
                    continue;
                }
            }

            if (!dataPoints.Any())
            {
                Console.WriteLine("Error: No valid data points created");
                return null;
            }

            var schemaDefinition = SchemaDefinition.Create(typeof(RegressionDataPoint));
            schemaDefinition[nameof(RegressionDataPoint.Features)].ColumnType = 
                new VectorDataViewType(NumberDataViewType.Single, featureColumns.Length);

            Console.WriteLine($"Created regression data view with {dataPoints.Count} rows and {featureColumns.Length} features");
            
            return mlContext.Data.LoadFromEnumerable(dataPoints, schemaDefinition);
        }

        #endregion

        #region Private Methods

        private bool CanAnalyzeData()
        {
            return _getDatabaseConfig != null && _getInputFields != null && !_isLoading;
        }

        private async void AnalyzeData()
        {
            try
            {
                IsLoading = true;
                LoadingMessage = "Validating configuration...";

                var databaseConfig = _getDatabaseConfig?.Invoke();
                var inputFields = _getInputFields?.Invoke();
                
                if (databaseConfig == null)
                {
                    _dialogService.ShowErrorDialog("Database configuration is not available.", "Error");
                    return;
                }

                if (inputFields == null || !inputFields.Any())
                {
                    _dialogService.ShowErrorDialog("No input fields are configured.", "Error");
                    return;
                }

                var enabledFields = inputFields.Where(f => f.IsEnabled).ToList();
                if (!enabledFields.Any())
                {
                    _dialogService.ShowErrorDialog("No input fields are enabled for analysis.", "Error");
                    return;
                }

                LoadingMessage = "Loading data from database...";
                await Task.Delay(200);

                var sqlHandler = new SqlHandler(databaseConfig.TableName);
                sqlHandler.Connect(databaseConfig);
                var connectionString = sqlHandler.GetConnectionString();

                var dataTable = await Task.Run(() => 
                {
                    var dataLoader = new DatabaseDataLoader();
                    var enabledFieldNames = enabledFields.Select(f => f.Name).ToArray();
                    
                    // Use same data loading approach as training but convert to DataTable for EDA
                    return LoadDataTableFromSql(
                        connectionString,
                        databaseConfig.TableName,
                        enabledFieldNames,
                        databaseConfig.WhereClause);
                });
                
                LoadingMessage = "Analyzing data patterns...";
                await Task.Delay(200);

                AnalyzeDataTable(dataTable);

                LoadingMessage = "Generating chart previews...";
                await Task.Delay(200);

                await _visualisationViewModel.GenerateHistogramPreviewsAsync(dataTable);

                LoadingMessage = "Setting up outlier detection...";
                await Task.Delay(100);

                var targetField = _getTargetField?.Invoke();
                _outlierDetectionViewModel.SetDataTable(dataTable, targetField);

                _dialogService.ShowInfoDialog($"Data analysis completed successfully for {enabledFields.Count} enabled fields.", "Analysis Complete");
            }
            catch (Exception ex)
            {
                _dialogService.ShowErrorDialog($"Error analyzing data: {ex.Message}", "Analysis Error");
            }
            finally
            {
                IsLoading = false;
            }
        }

        private DataTable LoadDataTableFromSql(string connectionString, string tableName, string[] fieldNames, string whereClause)
        {
            string fullTableName = tableName.Contains('[')
                ? tableName 
                : (tableName.Contains('.')
                    ? string.Join('.', tableName.Split('.').Select(part => $"[{part}]"))
                    : $"[{tableName}]");

            var fieldNamesWithBrackets = fieldNames.Select(f => $"[{f}]");
            var fieldsClause = string.Join(", ", fieldNamesWithBrackets);

            var query = string.IsNullOrWhiteSpace(whereClause) 
                ? $"SELECT {fieldsClause} FROM {fullTableName}"
                : $"SELECT {fieldsClause} FROM {fullTableName} WHERE {whereClause}";

            using var connection = new SqlConnection(connectionString);
            using var command = new SqlCommand(query, connection);
            using var adapter = new SqlDataAdapter(command);
            
            var dataTable = new DataTable();
            connection.Open();
            adapter.Fill(dataTable);
            
            return dataTable;
        }

        private void AnalyzeDataTable(DataTable dataTable)
        {
            NumberOfRows = dataTable.Rows.Count;
            NumberOfColumns = dataTable.Columns.Count;

            AnalyzeFeatureTypes(dataTable);

            AnalyzeMissingValues(dataTable);
        }

        private void AnalyzeFeatureTypes(DataTable dataTable)
        {
            var typeGroups = dataTable.Columns.Cast<DataColumn>()
                .GroupBy(col => GetFeatureType(col.DataType))
                .Select(g => new FeatureTypeInfo 
                { 
                    Type = g.Key, 
                    Count = g.Count() 
                })
                .OrderBy(x => x.Type);

            FeatureTypes.Clear();
            foreach (var typeInfo in typeGroups)
            {
                FeatureTypes.Add(typeInfo);
            }
        }

        private void AnalyzeMissingValues(DataTable dataTable)
        {
            var columnMissingInfo = new List<ColumnMissingInfo>();
            int totalCells = NumberOfRows * NumberOfColumns;
            int totalMissing = 0;

            foreach (DataColumn column in dataTable.Columns)
            {
                int missingCount = 0;
                
                foreach (DataRow row in dataTable.Rows)
                {
                    var value = row[column];
                    if (value == null || value == DBNull.Value || 
                        (value is string str && string.IsNullOrWhiteSpace(str)))
                    {
                        missingCount++;
                    }
                }

                totalMissing += missingCount;
                
                double missingPercentage = NumberOfRows > 0 ? (double)missingCount / NumberOfRows * 100 : 0;
                
                columnMissingInfo.Add(new ColumnMissingInfo
                {
                    ColumnName = column.ColumnName,
                    MissingCount = missingCount,
                    MissingPercentage = missingPercentage
                });
            }

            TotalMissingValues = totalMissing;
            MissingValuesPercentage = totalCells > 0 ? (double)totalMissing / totalCells * 100 : 0;

            ColumnMissingValues.Clear();
            foreach (var info in columnMissingInfo.OrderByDescending(x => x.MissingCount))
            {
                ColumnMissingValues.Add(info);
            }
        }

        private string GetFeatureType(Type dataType)
        {
            if (dataType == typeof(int) || dataType == typeof(long) || 
                dataType == typeof(short) || dataType == typeof(byte) ||
                dataType == typeof(float) || dataType == typeof(double) || 
                dataType == typeof(decimal))
            {
                return "Numeric";
            }
            else if (dataType == typeof(bool))
            {
                return "Boolean";
            }
            else if (dataType == typeof(DateTime))
            {
                return "Date";
            }
            else if (dataType == typeof(string))
            {
                return "Text/Categorical";
            }
            else
            {
                return "Other";
            }
        }

        #endregion
    }

    public class FeatureTypeInfo
    {
        public string Type { get; set; } = string.Empty;
        public int Count { get; set; }
    }

    public class ColumnMissingInfo
    {
        public string ColumnName { get; set; } = string.Empty;
        public int MissingCount { get; set; }
        public double MissingPercentage { get; set; }
    }

    // Data point classes for proper ML.NET conversion
    public class BinaryClassificationDataPoint
    {
        [VectorType]
        public float[] Features { get; set; } = Array.Empty<float>();
        
        public bool Label { get; set; }
    }

    public class MultiClassDataPoint
    {
        [VectorType]
        public float[] Features { get; set; } = Array.Empty<float>();
        
        public uint Label { get; set; }
    }

    public class RegressionDataPoint
    {
        [VectorType]
        public float[] Features { get; set; } = Array.Empty<float>();
        
        public float Label { get; set; }
    }
}