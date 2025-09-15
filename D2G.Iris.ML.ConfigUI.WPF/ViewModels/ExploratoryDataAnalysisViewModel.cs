using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using SciChart.Charting.Model.DataSeries;
using SciChart.Charting.Model.DataSeries.Heatmap2DArrayDataSeries;
using SciChart.Charting.Visuals;
using SciChart.Charting.Visuals.RenderableSeries;
using SciChart.Charting.Visuals.Axes;
using SciChart.Charting.Visuals.Axes.LabelProviders;
using SciChart.Charting.Themes;
using SciChart.Drawing.Common;
using SciChart.Charting.ChartModifiers;
using SciChart.Data.Model;
using SciChart.Core.Extensions;
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
    public class ExploratoryDataAnalysisViewModel : BaseViewModel, IDisposable
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
        private int _maxSampleSize = 10000;
        private bool _useDataSampling = false;
        private UserControl? _correlationHeatmapChart;
        private DataTable? _currentDataTable;

        public ExploratoryDataAnalysisViewModel(IDialogService dialogService)
        {
            _dialogService = dialogService;
            _featureTypes = new ObservableCollection<FeatureTypeInfo>();
            _columnMissingValues = new ObservableCollection<ColumnMissingInfo>();
            _visualisationViewModel = new VisualisationViewModel(dialogService);
            _outlierDetectionViewModel = new OutlierDetectionViewModel(dialogService);
            
            AnalyzeDataCommand = new RelayCommand(_ => AnalyzeData(), _ => CanAnalyzeData());
            GenerateCorrelationCommand = new RelayCommand(_ => GenerateCorrelationMatrix(), _ => CanGenerateCorrelation());
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

        public bool UseDataSampling
        {
            get => _useDataSampling;
            set => SetProperty(ref _useDataSampling, value);
        }

        public int MaxSampleSize
        {
            get => _maxSampleSize;
            set => SetProperty(ref _maxSampleSize, value);
        }

        public UserControl? CorrelationHeatmapChart
        {
            get => _correlationHeatmapChart;
            set => SetProperty(ref _correlationHeatmapChart, value);
        }

        #endregion

        #region Commands

        public ICommand AnalyzeDataCommand { get; }
        public ICommand GenerateCorrelationCommand { get; }

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
            var dataPoints = new List<BinaryClassificationDataPoint>(dataTable.Rows.Count);

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
                        var labelStr = labelValue.ToString()?.ToLower() ?? string.Empty;
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

            // Create schema definition
            var schemaDefinition = SchemaDefinition.Create(typeof(BinaryClassificationDataPoint));

            if (!dataPoints.Any())
            {
                Console.WriteLine("Error: No valid data points created");
                return mlContext.Data.LoadFromEnumerable(dataPoints, schemaDefinition);
            }
            schemaDefinition[nameof(BinaryClassificationDataPoint.Features)].ColumnType = 
                new VectorDataViewType(NumberDataViewType.Single, featureColumns.Length);

            Console.WriteLine($"Created binary classification data view with {dataPoints.Count} rows and {featureColumns.Length} features");
            
            return mlContext.Data.LoadFromEnumerable(dataPoints, schemaDefinition);
        }

        private IDataView CreateMultiClassDataView(MLContext mlContext, DataTable dataTable, string[] featureColumns, string targetColumn)
        {
            var dataPoints = new List<MultiClassDataPoint>(dataTable.Rows.Count);

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

            var schemaDefinition = SchemaDefinition.Create(typeof(MultiClassDataPoint));

            if (!dataPoints.Any())
            {
                Console.WriteLine("Error: No valid data points created");
                return mlContext.Data.LoadFromEnumerable(dataPoints, schemaDefinition);
            }
            schemaDefinition[nameof(MultiClassDataPoint.Features)].ColumnType = 
                new VectorDataViewType(NumberDataViewType.Single, featureColumns.Length);

            Console.WriteLine($"Created multi-class data view with {dataPoints.Count} rows and {featureColumns.Length} features");
            
            return mlContext.Data.LoadFromEnumerable(dataPoints, schemaDefinition);
        }

        private IDataView CreateRegressionDataView(MLContext mlContext, DataTable dataTable, string[] featureColumns, string targetColumn)
        {
            var dataPoints = new List<RegressionDataPoint>(dataTable.Rows.Count);

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

            var schemaDefinition = SchemaDefinition.Create(typeof(RegressionDataPoint));

            if (!dataPoints.Any())
            {
                Console.WriteLine("Error: No valid data points created");
                return mlContext.Data.LoadFromEnumerable(dataPoints, schemaDefinition);
            }
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

        private bool CanGenerateCorrelation()
        {
            return _currentDataTable != null && !_isLoading;
        }

        private async void GenerateCorrelationMatrix()
        {
            try
            {
                IsLoading = true;
                LoadingMessage = "Calculating correlations...";

                if (_currentDataTable == null)
                {
                    _dialogService.ShowErrorDialog("No data available. Please analyze data first.", "Error");
                    return;
                }

                // Calculate correlation matrix on background thread
                var correlationData = await Task.Run(() =>
                {
                    var numericColumns = GetNumericColumns(_currentDataTable);
                    if (numericColumns.Count < 2)
                    {
                        return (numericColumns, (double[,])null);
                    }
                    var correlationMatrix = CalculateCorrelationMatrix(_currentDataTable, numericColumns);
                    return (numericColumns, correlationMatrix);
                });

                // Create UI components on main thread
                UserControl correlationChart;
                if (correlationData.Item2 == null)
                {
                    correlationChart = CreateErrorControl("At least 2 numeric columns are required for correlation analysis.");
                }
                else
                {
                    correlationChart = CreateCorrelationHeatmap(correlationData.Item2, correlationData.numericColumns);
                }

                CorrelationHeatmapChart = correlationChart;

                _dialogService.ShowInfoDialog("Correlation matrix generated successfully.", "Correlation Analysis");
            }
            catch (Exception ex)
            {
                _dialogService.ShowErrorDialog($"Error generating correlation matrix: {ex.Message}", "Correlation Error");
            }
            finally
            {
                IsLoading = false;
            }
        }

        private UserControl CreateErrorControl(string message)
        {
            var errorControl = new UserControl();
            var textBlock = new TextBlock
            {
                Text = message,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                FontSize = 14,
                Foreground = Brushes.Gray,
                TextWrapping = TextWrapping.Wrap,
                Margin = new Thickness(20)
            };
            errorControl.Content = textBlock;
            return errorControl;
        }


        private List<string> GetNumericColumns(DataTable dataTable)
        {
            var numericColumns = new List<string>();

            foreach (DataColumn column in dataTable.Columns)
            {
                if (IsNumericType(column.DataType))
                {
                    numericColumns.Add(column.ColumnName);
                }
            }

            return numericColumns;
        }

        private bool IsNumericType(Type dataType)
        {
            return dataType == typeof(int) || dataType == typeof(long) ||
                   dataType == typeof(short) || dataType == typeof(byte) ||
                   dataType == typeof(float) || dataType == typeof(double) ||
                   dataType == typeof(decimal);
        }

        private double[,] CalculateCorrelationMatrix(DataTable dataTable, List<string> numericColumns)
        {
            int size = numericColumns.Count;
            var correlationMatrix = new double[size, size];

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    if (i == j)
                    {
                        correlationMatrix[i, j] = 1.0;
                    }
                    else
                    {
                        var correlation = CalculatePearsonCorrelation(
                            dataTable, numericColumns[i], numericColumns[j]);
                        correlationMatrix[i, j] = correlation;
                    }
                }
            }

            return correlationMatrix;
        }

        private double CalculatePearsonCorrelation(DataTable dataTable, string column1, string column2)
        {
            var values1 = new List<double>();
            var values2 = new List<double>();

            foreach (DataRow row in dataTable.Rows)
            {
                var val1 = row[column1];
                var val2 = row[column2];

                if (val1 != null && val1 != DBNull.Value &&
                    val2 != null && val2 != DBNull.Value &&
                    double.TryParse(val1.ToString(), out double d1) &&
                    double.TryParse(val2.ToString(), out double d2))
                {
                    values1.Add(d1);
                    values2.Add(d2);
                }
            }

            if (values1.Count < 2)
                return 0.0;

            double mean1 = values1.Average();
            double mean2 = values2.Average();

            double numerator = 0;
            double sumSq1 = 0;
            double sumSq2 = 0;

            for (int i = 0; i < values1.Count; i++)
            {
                double diff1 = values1[i] - mean1;
                double diff2 = values2[i] - mean2;

                numerator += diff1 * diff2;
                sumSq1 += diff1 * diff1;
                sumSq2 += diff2 * diff2;
            }

            double denominator = Math.Sqrt(sumSq1 * sumSq2);

            return denominator == 0 ? 0.0 : numerator / denominator;
        }

        public class FeatureNameLabelProvider : LabelProviderBase
        {
            private readonly string[] featureNames;

            public FeatureNameLabelProvider(string[] featureNames)
            {
                this.featureNames = featureNames;
            }

            public override string FormatLabel(IComparable dataValue)
            {
                try
                {
                    // Map tick at k-0.5 to index k
                    int index = (int)Math.Floor(Convert.ToDouble(dataValue) + 0.5);
                    if (index >= 0 && index < featureNames.Length)
                    {
                        // Truncate long feature names for better display
                        string name = featureNames[index];
                        return name.Length > 12 ? name.Substring(0, 12) + "..." : name;
                    }
                    return string.Empty;
                }
                catch
                {
                    return string.Empty;
                }
            }

            public override string FormatCursorLabel(IComparable dataValue)
            {
                try
                {
                    int index = (int)Math.Floor(Convert.ToDouble(dataValue) + 0.5);
                    if (index >= 0 && index < featureNames.Length)
                        return featureNames[index]; // Full name in cursor tooltip
                    return string.Empty;
                }
                catch
                {
                    return string.Empty;
                }
            }
        }

        // For Y-Axis when FlipCoordinates = true to align names with grid rows top-to-bottom
        public class ReversedFeatureNameLabelProvider : LabelProviderBase
        {
            private readonly string[] featureNames;

            public ReversedFeatureNameLabelProvider(string[] featureNames)
            {
                this.featureNames = featureNames;
            }

            public override string FormatLabel(IComparable dataValue)
            {
                try
                {
                    int index = (int)Math.Floor(Convert.ToDouble(dataValue) + 0.5);
                    if (index >= 0 && index < featureNames.Length)
                    {
                        int reversedIndex = featureNames.Length - 1 - index;
                        string name = featureNames[reversedIndex];
                        return name.Length > 12 ? name.Substring(0, 12) + "..." : name;
                    }
                    return string.Empty;
                }
                catch
                {
                    return string.Empty;
                }
            }

            public override string FormatCursorLabel(IComparable dataValue)
            {
                try
                {
                    int index = (int)Math.Floor(Convert.ToDouble(dataValue) + 0.5);
                    if (index >= 0 && index < featureNames.Length)
                    {
                        int reversedIndex = featureNames.Length - 1 - index;
                        return featureNames[reversedIndex];
                    }
                    return string.Empty;
                }
                catch
                {
                    return string.Empty;
                }
            }
        }

        private UserControl CreateCorrelationHeatmap(double[,] correlationMatrix, List<string> columnNames)
        {
            int size = columnNames.Count;

            var containerControl = new UserControl();
            var mainGrid = new Grid();

            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

            var titleBlock = new TextBlock
            {
                Text = "Correlation Heatmap",
                FontSize = 16,
                FontWeight = FontWeights.Bold,
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(10)
            };
            Grid.SetRow(titleBlock, 0);
            mainGrid.Children.Add(titleBlock);

            // Create SciChart heatmap with optimal size for zoom interactions
            var sciChartSurface = new SciChartSurface
            {
                Height = 650,
                Width = 900,
                Margin = new Thickness(5),
                Background = Brushes.White,
                Padding = new Thickness(10)
            };

            // Create heatmap data series using the correct API from SciChart example
            var heatmapDataSeries = new UniformHeatmapDataSeries<int, int, double>(correlationMatrix, 0, 1, 0, 1);

            // Create a heatmap renderable series using FastUniformHeatmapRenderableSeries
            var heatmapSeries = new FastUniformHeatmapRenderableSeries
            {
                DataSeries = heatmapDataSeries,
                DrawTextInCell = true,
                Opacity = 1.0
            };

            // Create color map for correlation values (-1 to +1)
            var colorMap = new HeatmapColorPalette
            {
                Minimum = -1.0,
                Maximum = 1.0
            };

            // Add gradient stops for correlation visualization
            colorMap.GradientStops.Add(new GradientStop(Colors.Blue, 0.0));    // -1 (strong negative)
            colorMap.GradientStops.Add(new GradientStop(Colors.Cyan, 0.25));   // -0.5
            colorMap.GradientStops.Add(new GradientStop(Colors.White, 0.5));   // 0 (no correlation)
            colorMap.GradientStops.Add(new GradientStop(Colors.Yellow, 0.75)); // +0.5
            colorMap.GradientStops.Add(new GradientStop(Colors.Red, 1.0));     // +1 (strong positive)

            heatmapSeries.ColorMap = colorMap;

            // Convert List<string> to string[] for the FeatureNameLabelProvider
            string[] featureNames = columnNames.ToArray();

            // Configure axes with custom label providers to show feature names against each grid cell (centered)
            var xAxis = new NumericAxis
            {
                AxisTitle = "Features",
                VisibleRange = new DoubleRange(-0.5, size - 0.5),
                MajorDelta = 1,
                MinorDelta = 1,
                DrawMinorTicks = false,
                DrawMajorTicks = true,
                DrawMajorGridLines = true,  // Enable grid lines to show feature separation
                DrawMinorGridLines = false,
                DrawMajorBands = false,
                AutoTicks = false,
                LabelProvider = new FeatureNameLabelProvider(featureNames),
                AxisAlignment = AxisAlignment.Bottom
            };

            var yAxis = new NumericAxis
            {
                AxisTitle = "Features",
                VisibleRange = new DoubleRange(-0.5, size - 0.5),
                MajorDelta = 1,
                MinorDelta = 1,
                DrawMinorTicks = false,
                DrawMajorTicks = true,
                DrawMajorGridLines = true,  // Enable grid lines to show feature separation
                DrawMinorGridLines = false,
                DrawMajorBands = false,
                AutoTicks = false,
                // Use reversed provider to match flipped Y-axis so first feature is at the top row
                LabelProvider = new ReversedFeatureNameLabelProvider(featureNames),
                AxisAlignment = AxisAlignment.Left,
                FlipCoordinates = true
            };

            sciChartSurface.XAxes.Add(xAxis);
            sciChartSurface.YAxes.Add(yAxis);
            sciChartSurface.RenderableSeries.Add(heatmapSeries);

            // Add interactive chart modifiers for zoom, pan, and mouse interactions
            sciChartSurface.ChartModifier = new ModifierGroup(
                // Mouse wheel zoom
                new MouseWheelZoomModifier(),
                // Rubber band zoom (drag to select area to zoom)
                new RubberBandXyZoomModifier(),
                // Double-click to zoom to extents
                new ZoomExtentsModifier(),
                // Pan with right mouse button drag
                new ZoomPanModifier { ExecuteOn = ExecuteOn.MouseRightButton },
                // Cursor modifier to show crosshair and values
                new CursorModifier { ShowTooltip = true, ShowAxisLabels = true },
                // Allow dragging of axes for fine adjustment
                new XAxisDragModifier(),
                new YAxisDragModifier()
            );

            Grid.SetRow(sciChartSurface, 1);
            mainGrid.Children.Add(sciChartSurface);

            var legendPanel = CreateLegendPanel();
            Grid.SetRow(legendPanel, 2);
            mainGrid.Children.Add(legendPanel);

            var instructionText = new TextBlock
            {
                Text = "Mouse Controls: Wheel=Zoom | Left Drag=Select Zoom Area | Right Drag=Pan | Double Click=Fit to View | Hover=Show Values",
                TextWrapping = TextWrapping.Wrap,
                FontSize = 11,
                Foreground = Brushes.Gray,
                FontStyle = FontStyles.Italic,
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(10, 5, 10, 10)
            };

            Grid.SetRow(instructionText, 3);
            mainGrid.Children.Add(instructionText);

            containerControl.Content = mainGrid;
            return containerControl;
        }

        private StackPanel CreateLegendPanel()
        {
            var legendPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(10)
            };

            legendPanel.Children.Add(new TextBlock
            {
                Text = "Legend: ",
                FontWeight = FontWeights.Bold,
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 10, 0)
            });

            // SciChart default heatmap colors (blue to red gradient)
            var legendItems = new[]
            {
                (Colors.Blue, "-1 (Strong Negative)"),
                (Colors.Cyan, "-0.5 (Negative)"),
                (Colors.Yellow, "0 (No Correlation)"),
                (Colors.Orange, "0.5 (Positive)"),
                (Colors.Red, "+1 (Strong Positive)")
            };

            foreach (var (color, description) in legendItems)
            {
                legendPanel.Children.Add(new Rectangle
                {
                    Width = 20,
                    Height = 15,
                    Fill = new SolidColorBrush(color),
                    Margin = new Thickness(0, 0, 5, 0)
                });

                legendPanel.Children.Add(new TextBlock
                {
                    Text = description,
                    VerticalAlignment = VerticalAlignment.Center,
                    Margin = new Thickness(0, 0, 15, 0),
                    FontSize = 10
                });
            }

            return legendPanel;
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

                // Find this section in AnalyzeData():
                var dataTable = await Task.Run(() =>
                {
                    var dataLoader = new DatabaseDataLoader();
                    var enabledFieldNames = enabledFields.Select(f => f.Name).ToArray();

                    // ADD THESE LINES:
                    var targetField = _getTargetField?.Invoke();
                    var allFieldsForEDA = enabledFieldNames.ToList();
                    if (!string.IsNullOrEmpty(targetField) && !allFieldsForEDA.Contains(targetField))
                    {
                        allFieldsForEDA.Add(targetField);
                    }

                    // CHANGE THIS LINE:
                    return LoadDataTableFromSql(
                        connectionString,
                        databaseConfig.TableName,
                        allFieldsForEDA.ToArray(), // Changed from enabledFieldNames
                        databaseConfig.WhereClause);
                });

                LoadingMessage = "Analyzing data patterns...";
                await Task.Delay(200);

                AnalyzeDataTable(dataTable);

                _currentDataTable = dataTable;

                LoadingMessage = "Setting up outlier detection...";
                await Task.Delay(100);
                
                var targetField = _getTargetField?.Invoke();
                _outlierDetectionViewModel.SetDataTable(dataTable, targetField);

                LoadingMessage = "Preparing visualization data...";
                await Task.Delay(50);

                // Create data table without target column for visualization
                var visualizationDataTable = await Task.Run(() => CreateDataTableWithoutTarget(dataTable, targetField));
                _visualisationViewModel.SetDataTable(visualizationDataTable);

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

        private DataTable CreateSampledDataTable(DataTable originalTable, string? targetField)
        {
            var sampleSize = Math.Min(_maxSampleSize, originalTable.Rows.Count);
            var step = Math.Max(1, originalTable.Rows.Count / sampleSize);

            // Create new table with sampled data
            var sampledTable = new DataTable();

            // Add all columns except target
            foreach (DataColumn column in originalTable.Columns)
            {
                if (string.IsNullOrEmpty(targetField) || column.ColumnName != targetField)
                {
                    sampledTable.Columns.Add(column.ColumnName, column.DataType);
                }
            }

            // Sample rows
            for (int i = 0; i < originalTable.Rows.Count && sampledTable.Rows.Count < sampleSize; i += step)
            {
                var originalRow = originalTable.Rows[i];
                var newRow = sampledTable.NewRow();

                foreach (DataColumn column in sampledTable.Columns)
                {
                    newRow[column.ColumnName] = originalRow[column.ColumnName];
                }

                sampledTable.Rows.Add(newRow);
            }

            return sampledTable;
        }

        private DataTable CreateDataTableWithoutTarget(DataTable originalTable, string? targetField)
        {
            // If no target field or target field doesn't exist, return the original table
            if (string.IsNullOrEmpty(targetField) || !originalTable.Columns.Contains(targetField))
            {
                return originalTable;
            }

            // For large datasets, use sampling to reduce memory usage
            if (_useDataSampling && originalTable.Rows.Count > _maxSampleSize)
            {
                return CreateSampledDataTable(originalTable, targetField);
            }

            // Create a new DataTable with all columns except the target
            var filteredTable = new DataTable();

            // Add all columns except the target column
            foreach (DataColumn column in originalTable.Columns)
            {
                if (column.ColumnName != targetField)
                {
                    filteredTable.Columns.Add(column.ColumnName, column.DataType);
                }
            }

            // Get the indices of columns to copy (excluding target column)
            var columnIndices = new List<int>();

            for (int i = 0; i < originalTable.Columns.Count; i++)
            {
                if (originalTable.Columns[i].ColumnName != targetField)
                {
                    columnIndices.Add(i);
                }
            }

            // Batch copy data using ItemArray for better performance
            foreach (DataRow originalRow in originalTable.Rows)
            {
                var newRow = filteredTable.NewRow();
                var originalItems = originalRow.ItemArray;

                for (int i = 0; i < columnIndices.Count; i++)
                {
                    newRow[i] = originalItems[columnIndices[i]];
                }

                filteredTable.Rows.Add(newRow);
            }

            return filteredTable;
        }


        public void ClearMemoryCache()
        {
            // Clear collections to free memory
            _featureTypes?.Clear();
            _columnMissingValues?.Clear();

            // Force garbage collection if needed for large datasets
            if (_useDataSampling)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        public new void Dispose()
        {
            ClearMemoryCache();
            _visualisationViewModel = null!;
            _outlierDetectionViewModel = null!;
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
