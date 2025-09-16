using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Data;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using SciChart.Charting.Model.ChartSeries;
using SciChart.Charting.Model.DataSeries;
using SciChart.Data.Model;

using D2G.Iris.ML.ConfigUI.WPF.Services;
using D2G.Iris.ML.ConfigUI.WPF.Commands;

namespace D2G.Iris.ML.ConfigUI.WPF.ViewModels
{
    public class VisualisationViewModel : INotifyPropertyChanged
    {
        private readonly IDialogService _dialogService;
        private ObservableCollection<HistogramViewModel> _histograms;
        private HistogramViewModel? _selectedHistogram;
        private bool _isDetailViewVisible;
        private bool _isGeneratingHistograms;
        private string _progressMessage = string.Empty;
        private CancellationTokenSource? _cancellationTokenSource;
        private DataTable? _dataTable;
        private string _dataInfo = string.Empty;
        private readonly int _chunkSize = 1000;

        public VisualisationViewModel(IDialogService dialogService)
        {
            _dialogService = dialogService;
            _histograms = new ObservableCollection<HistogramViewModel>();
            
            SelectHistogramCommand = new AsyncRelayCommand(async obj => await SelectHistogramAsync(obj as HistogramViewModel));
            BackToOverviewCommand = new RelayCommand(_ => BackToOverview());
            CancelGenerationCommand = new RelayCommand(_ => CancelGeneration());
            GenerateHistogramsCommand = new AsyncRelayCommand(async _ => await GenerateHistogramsAsync(), _ => CanGenerateHistograms());
        }

        public ObservableCollection<HistogramViewModel> Histograms
        {
            get => _histograms;
            set => SetProperty(ref _histograms, value);
        }

        public HistogramViewModel? SelectedHistogram
        {
            get => _selectedHistogram;
            set => SetProperty(ref _selectedHistogram, value);
        }

        public bool IsDetailViewVisible
        {
            get => _isDetailViewVisible;
            set => SetProperty(ref _isDetailViewVisible, value);
        }

        public ICommand SelectHistogramCommand { get; }
        public ICommand BackToOverviewCommand { get; }
        public ICommand CancelGenerationCommand { get; }
        public ICommand GenerateHistogramsCommand { get; }

        public string DataInfo
        {
            get => _dataInfo;
            set => SetProperty(ref _dataInfo, value);
        }

        public bool IsGeneratingHistograms
        {
            get => _isGeneratingHistograms;
            set => SetProperty(ref _isGeneratingHistograms, value);
        }

        public string ProgressMessage
        {
            get => _progressMessage;
            set => SetProperty(ref _progressMessage, value);
        }

        public async Task GenerateHistogramPreviewsAsync(DataTable dataTable)
        {
            _dataTable = dataTable;
            _cancellationTokenSource?.Cancel();
            _cancellationTokenSource = new CancellationTokenSource();
            var cancellationToken = _cancellationTokenSource.Token;

            IsGeneratingHistograms = true;
            Histograms.Clear();

            try
            {
                ProgressMessage = "Analyzing columns...";
                var columns = dataTable.Columns.Cast<DataColumn>()
                    .Where(c => IsNumericColumn(c) || c.DataType == typeof(string))
                    .ToList();

                int processedColumns = 0;
                int totalColumns = columns.Count;

                foreach (DataColumn column in columns)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    
                    ProgressMessage = $"Creating preview for {column.ColumnName} ({processedColumns + 1}/{totalColumns})...";

                    var preview = CreateHistogramPreview(column, dataTable);
                    if (preview != null)
                    {
                        Histograms.Add(preview);
                    }

                    processedColumns++;
                    await Task.Delay(5, cancellationToken); 
                }

                ProgressMessage = "Previews ready";
                await Task.Delay(50, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                
            }
            catch (Exception ex)
            {
                _dialogService.ShowErrorDialog($"Error generating previews: {ex.Message}", "Error");
            }
            finally
            {
                IsGeneratingHistograms = false;
                ProgressMessage = string.Empty;
                _cancellationTokenSource?.Dispose();
                _cancellationTokenSource = null;
            }
        }

        public async Task GenerateHistogramsAsync(DataTable dataTable)
        {
            SetDataTable(dataTable);
            await GenerateHistogramPreviewsAsync(dataTable);
        }

        public async Task GenerateHistogramsAsync()
        {
            if (_dataTable != null)
            {
                
                await GenerateHistogramPreviewsAsync(_dataTable);
            }
            else
            {
                _dialogService.ShowErrorDialog("No data available for visualization. Please run data analysis first.", "No Data");
            }
        }

        
        public void GenerateHistograms(DataTable dataTable)
        {
            SetDataTable(dataTable);
            _ = GenerateHistogramPreviewsAsync(dataTable);
        }

        public void SetDataTable(DataTable dataTable)
        {
            _dataTable = dataTable;
            UpdateDataInfo();
        }


        private bool CanGenerateHistograms()
        {
            return _dataTable != null && !IsGeneratingHistograms;
        }

        private void UpdateDataInfo()
        {
            if (_dataTable != null)
            {
                var numericColumns = _dataTable.Columns.Cast<DataColumn>()
                    .Count(col => IsNumericColumn(col));
                DataInfo = $"Data ready: {_dataTable.Rows.Count} rows, {numericColumns} numeric columns";
            }
            else
            {
                DataInfo = "No data available - run data analysis first";
            }
        }

        private HistogramViewModel CreateHistogramPreview(DataColumn column, DataTable dataTable)
        {
            var totalRows = dataTable.Rows.Count;
            var columnType = IsNumericColumn(column) ? "Numeric" : "Categorical";
            
            
            if (IsNumericColumn(column))
            {
                return CreateSimpleNumericPreview(column, dataTable);
            }
            else if (column.DataType == typeof(string))
            {
                return CreateSimpleCategoricalPreview(column, dataTable);
            }
            
            
            var sampleSize = Math.Min(100, totalRows);
            var dataSeries = new XyDataSeries<double, int>();
            dataSeries.Append(0, 0); 
            
            return new HistogramViewModel
            {
                ColumnName = column.ColumnName,
                ColumnType = columnType,
                TotalCount = totalRows,
                IsPreviewOnly = true,
                DataSeries = dataSeries,
                PreviewInfo = new PreviewInfo
                {
                    SampleSize = sampleSize,
                    NonNullCount = 0,
                    UniqueValueCount = 0,
                    MissingCount = sampleSize
                }
            };
        }

        private HistogramViewModel CreateSimpleNumericPreview(DataColumn column, DataTable dataTable)
        {
            var values = new List<double>();
            var totalRows = dataTable.Rows.Count;

            
            for (int i = 0; i < totalRows; i++)
            {
                var value = dataTable.Rows[i][column];
                if (value != null && value != DBNull.Value && double.TryParse(value.ToString(), out double numericValue))
                {
                    values.Add(numericValue);
                }
            }
            
            if (!values.Any())
            {
                var emptyDataSeries = new XyDataSeries<double, int>();
                emptyDataSeries.Append(0, 0);
                return new HistogramViewModel
                {
                    ColumnName = column.ColumnName,
                    ColumnType = "Numeric",
                    TotalCount = dataTable.Rows.Count,
                    IsPreviewOnly = true,
                    DataSeries = emptyDataSeries
                };
            }
            
            
            var bins = 10;
            var min = values.Min();
            var max = values.Max();
            var range = max - min;
            
            var dataSeries = new XyDataSeries<double, int>();
            
            if (range == 0)
            {
                dataSeries.Append(min, values.Count);
            }
            else
            {
                var binWidth = range / bins;
                for (int i = 0; i < bins; i++)
                {
                    var binStart = min + i * binWidth;
                    var binEnd = binStart + binWidth;
                    var count = values.Count(v => v >= binStart && (i == bins - 1 ? v <= binEnd : v < binEnd));
                    var binCenter = binStart + binWidth / 2;
                    dataSeries.Append(binCenter, count);
                }
            }
            
            return new HistogramViewModel
            {
                ColumnName = column.ColumnName,
                ColumnType = "Numeric",
                TotalCount = dataTable.Rows.Count,
                IsPreviewOnly = false,
                DataSeries = dataSeries,
                PreviewInfo = new PreviewInfo
                {
                    SampleSize = totalRows,
                    NonNullCount = values.Count,
                    UniqueValueCount = values.Distinct().Count(),
                    MissingCount = totalRows - values.Count
                }
            };
        }

        private HistogramViewModel CreateSimpleCategoricalPreview(DataColumn column, DataTable dataTable)
        {
            var allValues = new List<string>();
            var totalRows = dataTable.Rows.Count;

            
            for (int i = 0; i < totalRows; i++)
            {
                var value = dataTable.Rows[i][column]?.ToString();
                if (!string.IsNullOrWhiteSpace(value))
                {
                    allValues.Add(value);
                }
            }
            
            var categoryGroups = allValues
                .GroupBy(s => s)
                .OrderByDescending(g => g.Count())
                .Take(20) 
                .ToList();
            
            var dataSeries = new XyDataSeries<double, int>();
            
            if (!categoryGroups.Any())
            {
                dataSeries.Append(0, 0);
            }
            else
            {
                for (int i = 0; i < categoryGroups.Count; i++)
                {
                    dataSeries.Append(i + 0.5, categoryGroups[i].Count());
                }
            }
            
            return new HistogramViewModel
            {
                ColumnName = column.ColumnName,
                ColumnType = "Categorical",
                TotalCount = dataTable.Rows.Count,
                IsPreviewOnly = false,
                DataSeries = dataSeries,
                PreviewInfo = new PreviewInfo
                {
                    SampleSize = totalRows,
                    NonNullCount = allValues.Count,
                    UniqueValueCount = allValues.Distinct().Count(),
                    MissingCount = totalRows - allValues.Count
                }
            };
        }

        public async Task LoadFullHistogramAsync(HistogramViewModel histogram)
        {
            if (_dataTable == null || histogram.IsLoading) return;
            
            try
            {
                histogram.IsLoading = true;
                
                
                if (!_dataTable.Columns.Contains(histogram.ColumnName))
                {
                    _dialogService.ShowErrorDialog($"Column '{histogram.ColumnName}' no longer exists in the dataset.", "Error");
                    return;
                }
                
                var column = _dataTable.Columns[histogram.ColumnName];
                if (column == null)
                {
                    _dialogService.ShowErrorDialog($"Failed to access column '{histogram.ColumnName}'.", "Error");
                    return;
                }
                
                
                HistogramViewModel? detailedHistogram = null;
                
                if (IsNumericColumn(column))
                {
                    detailedHistogram = await CreateNumericHistogramAsync(column, _dataTable, CancellationToken.None);
                }
                else if (column.DataType == typeof(string))
                {
                    detailedHistogram = await CreateCategoricalHistogramAsync(column, _dataTable, CancellationToken.None);
                }
                
                if (detailedHistogram != null)
                {
                    
                    detailedHistogram.IsSelected = histogram.IsSelected;
                    
                    var index = Histograms.IndexOf(histogram);
                    if (index >= 0 && index < Histograms.Count)
                    {
                        Histograms[index] = detailedHistogram;
                    }
                }
            }
            catch (Exception ex)
            {
                _dialogService.ShowErrorDialog($"Error loading detailed histogram for {histogram.ColumnName}: {ex.Message}", "Error");
            }
            finally
            {
                histogram.IsLoading = false;
            }
        }


        private bool IsNumericColumn(DataColumn column)
        {
            return column.DataType == typeof(int) || 
                   column.DataType == typeof(long) || 
                   column.DataType == typeof(short) || 
                   column.DataType == typeof(byte) ||
                   column.DataType == typeof(float) || 
                   column.DataType == typeof(double) || 
                   column.DataType == typeof(decimal);
        }

        private async Task<HistogramViewModel?> CreateNumericHistogramAsync(DataColumn column, DataTable dataTable, CancellationToken cancellationToken)
        {
            if (column == null || dataTable == null) return null;
            return await Task.Run(() => CreateNumericHistogram(column, dataTable), cancellationToken);
        }

        private HistogramViewModel? CreateNumericHistogram(DataColumn column, DataTable dataTable)
        {
            if (column == null || dataTable == null) return null;
            
            var values = new List<double>();
            var allValues = new List<object?>();
            
            
            int rowCount = dataTable.Rows.Count;

            for (int i = 0; i < rowCount; i++)
            {
                var value = dataTable.Rows[i][column];
                allValues.Add(value);

                if (value != null && value != DBNull.Value)
                {
                    if (double.TryParse(value.ToString(), out double numericValue))
                    {
                        values.Add(numericValue);
                    }
                }

                
                if (i % _chunkSize == 0)
                {
                    Thread.Yield();
                }
            }

            if (!values.Any()) return null;

            var bins = 20;
            var min = values.Min();
            var max = values.Max();
            var range = max - min;
            
            var histogram = new List<HistogramBin>();

            
            if (range == 0 || Math.Abs(range) < double.Epsilon)
            {
                
                
                var totalCount = values.Count;
                var spread = Math.Max(1.0, Math.Abs(min) * 0.1); 
                if (spread == 0) spread = 1.0; 
                
                histogram.Add(new HistogramBin
                {
                    BinStart = min - spread,
                    BinEnd = min - spread/3,
                    Count = 0,
                    BinCenter = min - spread * 2/3
                });
                
                histogram.Add(new HistogramBin
                {
                    BinStart = min - spread/3,
                    BinEnd = min + spread/3,
                    Count = totalCount,
                    BinCenter = min,
                    CategoryName = min.ToString("F2")
                });
                
                histogram.Add(new HistogramBin
                {
                    BinStart = min + spread/3,
                    BinEnd = min + spread,
                    Count = 0,
                    BinCenter = min + spread * 2/3
                });
            }
            else
            {
                var binWidth = range / bins;
                
                for (int i = 0; i < bins; i++)
                {
                    var binStart = min + i * binWidth;
                    var binEnd = binStart + binWidth;
                    var count = values.Count(v => v >= binStart && (i == bins - 1 ? v <= binEnd : v < binEnd));
                    
                    histogram.Add(new HistogramBin
                    {
                        BinStart = binStart,
                        BinEnd = binEnd,
                        Count = count,
                        BinCenter = binStart + binWidth / 2
                    });
                }
            }

            var dataSeries = new XyDataSeries<double, int>();
            foreach (var bin in histogram)
            {
                dataSeries.Append(bin.BinCenter, bin.Count);
            }

            var statistics = CalculateNumericStatistics(values, allValues);

            var result = new HistogramViewModel
            {
                ColumnName = column.ColumnName,
                ColumnType = "Numeric",
                Bins = histogram,
                TotalCount = values.Count,
                DataSeries = dataSeries,
                Statistics = statistics
            };
            
            return result;
        }

        private async Task<HistogramViewModel?> CreateCategoricalHistogramAsync(DataColumn column, DataTable dataTable, CancellationToken cancellationToken)
        {
            if (column == null || dataTable == null) return null;
            return await Task.Run(() => CreateCategoricalHistogram(column, dataTable), cancellationToken);
        }

        private HistogramViewModel? CreateCategoricalHistogram(DataColumn column, DataTable dataTable)
        {
            if (column == null || dataTable == null) return null;
            
            var allValues = new List<string?>();
            
            
            int rowCount = dataTable.Rows.Count;

            for (int i = 0; i < rowCount; i++)
            {
                allValues.Add(dataTable.Rows[i][column]?.ToString());

                
                if (i % _chunkSize == 0)
                {
                    Thread.Yield();
                }
            }

            var categoryGroups = allValues
                .Where(s => !string.IsNullOrWhiteSpace(s))
                .GroupBy(s => s!)
                .OrderByDescending(g => g.Count())
                .Take(20)
                .ToList();

            if (!categoryGroups.Any()) return null;

            var histogram = categoryGroups.Select((group, index) => new HistogramBin
            {
                BinStart = index,
                BinEnd = index + 1,
                Count = group.Count(),
                BinCenter = index + 0.5,
                CategoryName = group.Key
            }).ToList();

            var dataSeries = new XyDataSeries<double, int>();
            foreach (var bin in histogram)
            {
                dataSeries.Append(bin.BinCenter, bin.Count);
            }

            var statistics = CalculateCategoricalStatistics(allValues);

            return new HistogramViewModel
            {
                ColumnName = column.ColumnName,
                ColumnType = "Categorical",
                Bins = histogram,
                TotalCount = categoryGroups.Sum(g => g.Count()),
                DataSeries = dataSeries,
                Statistics = statistics
            };
        }

        private StatisticalSummary CalculateNumericStatistics(List<double> values, List<object?> allValues)
        {
            if (!values.Any()) return new StatisticalSummary();

            var sortedValues = values.OrderBy(x => x).ToList();
            var n = values.Count;
            
            var mean = values.Average();
            var variance = n > 1 ? values.Sum(x => Math.Pow(x - mean, 2)) / n : 0;
            var standardDeviation = Math.Sqrt(Math.Max(0, variance)); 
            
            var median = n % 2 == 0 
                ? (sortedValues[n / 2 - 1] + sortedValues[n / 2]) / 2.0
                : sortedValues[n / 2];
            
            var q1 = CalculatePercentile(sortedValues, 25);
            var q3 = CalculatePercentile(sortedValues, 75);
            var iqr = Math.Max(0, q3 - q1); 
            
            var skewness = CalculateSkewness(values, mean, standardDeviation);
            var kurtosis = CalculateKurtosis(values, mean, standardDeviation);
            
            var missingCount = allValues.Count(v => v == null || v == DBNull.Value || 
                (v is string str && string.IsNullOrWhiteSpace(str)));

            var frequencies = values.GroupBy(x => x).OrderByDescending(g => g.Count()).FirstOrDefault();
            var range = sortedValues.Count > 1 ? sortedValues.Last() - sortedValues.First() : 0;

            return new StatisticalSummary
            {
                Mean = Double.IsNaN(mean) ? 0 : mean,
                Median = Double.IsNaN(median) ? 0 : median,
                StandardDeviation = Double.IsNaN(standardDeviation) ? 0 : standardDeviation,
                Variance = Double.IsNaN(variance) ? 0 : variance,
                Min = sortedValues.First(),
                Max = sortedValues.Last(),
                Range = range,
                Q1 = Double.IsNaN(q1) ? sortedValues.First() : q1,
                Q3 = Double.IsNaN(q3) ? sortedValues.Last() : q3,
                IQR = Double.IsNaN(iqr) ? 0 : iqr,
                Skewness = Double.IsNaN(skewness) ? 0 : skewness,
                Kurtosis = Double.IsNaN(kurtosis) ? 0 : kurtosis,
                UniqueValues = values.Distinct().Count(),
                MissingValues = missingCount,
                MostFrequentValue = frequencies?.Key.ToString() ?? "",
                MostFrequentCount = frequencies?.Count() ?? 0
            };
        }

        private StatisticalSummary CalculateCategoricalStatistics(List<string?> allValues)
        {
            var nonNullValues = allValues.Where(v => !string.IsNullOrWhiteSpace(v)).ToList();
            var missingCount = allValues.Count - nonNullValues.Count;
            
            var frequencies = nonNullValues.GroupBy(x => x).OrderByDescending(g => g.Count()).ToList();
            var mostFrequent = frequencies.FirstOrDefault();

            return new StatisticalSummary
            {
                UniqueValues = nonNullValues.Distinct().Count(),
                MissingValues = missingCount,
                MostFrequentValue = mostFrequent?.Key ?? "",
                MostFrequentCount = mostFrequent?.Count() ?? 0
            };
        }

        private double CalculatePercentile(List<double> sortedValues, double percentile)
        {
            if (sortedValues == null || !sortedValues.Any())
                return 0;
                
            var n = sortedValues.Count;
            
            if (n == 1)
                return sortedValues[0];
                
            var index = percentile / 100.0 * (n - 1);
            
            if (index <= 0)
                return sortedValues[0];
            if (index >= n - 1)
                return sortedValues[n - 1];
            
            if (index == Math.Floor(index))
            {
                return sortedValues[(int)index];
            }
            else
            {
                var lower = (int)Math.Floor(index);
                var upper = (int)Math.Ceiling(index);
                var weight = index - lower;
                
                
                lower = Math.Max(0, Math.Min(lower, n - 1));
                upper = Math.Max(0, Math.Min(upper, n - 1));
                
                return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
            }
        }

        private double CalculateSkewness(List<double> values, double mean, double standardDeviation)
        {
            if (standardDeviation == 0) return 0;
            
            var n = values.Count;
            var sum = values.Sum(x => Math.Pow((x - mean) / standardDeviation, 3));
            
            return sum / n;
        }

        private double CalculateKurtosis(List<double> values, double mean, double standardDeviation)
        {
            if (standardDeviation == 0) return 0;
            
            var n = values.Count;
            var sum = values.Sum(x => Math.Pow((x - mean) / standardDeviation, 4));
            
            return (sum / n) - 3; 
        }

        private async Task SelectHistogramAsync(HistogramViewModel? histogram)
        {
            if (histogram == null) return;
            
            try
            {
                foreach (var h in Histograms)
                {
                    h.IsSelected = false;
                }
                
                histogram.IsSelected = true;
                
                
                await LoadFullHistogramAsync(histogram);
                
                histogram = Histograms.FirstOrDefault(h => h.ColumnName == histogram.ColumnName);
                if (histogram == null)
                {
                    _dialogService.ShowErrorDialog("Failed to load histogram data.", "Error");
                    return;
                }
                
                SelectedHistogram = histogram;
                IsDetailViewVisible = true;
            }
            catch (Exception ex)
            {
                _dialogService.ShowErrorDialog($"Error selecting histogram: {ex.Message}", "Error");
            }
        }

        private void BackToOverview()
        {
            IsDetailViewVisible = false;
            if (SelectedHistogram != null)
            {
                SelectedHistogram.IsSelected = false;
            }
            SelectedHistogram = null;
        }

        private void CancelGeneration()
        {
            _cancellationTokenSource?.Cancel();
        }


        public event PropertyChangedEventHandler? PropertyChanged;
        
        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected bool SetProperty<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return false;
            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }
    }

    public class HistogramViewModel : INotifyPropertyChanged
    {
        private bool _isSelected;
        private bool _isLoading;

        public string ColumnName { get; set; } = string.Empty;
        public string ColumnType { get; set; } = string.Empty;
        public List<HistogramBin> Bins { get; set; } = new();
        public int TotalCount { get; set; }
        public IDataSeries DataSeries { get; set; } = null!;
        public StatisticalSummary Statistics { get; set; } = new();
        public bool IsPreviewOnly { get; set; } = false;
        public PreviewInfo PreviewInfo { get; set; } = new();

        public bool IsSelected
        {
            get => _isSelected;
            set => SetProperty(ref _isSelected, value);
        }

        public bool IsLoading
        {
            get => _isLoading;
            set => SetProperty(ref _isLoading, value);
        }

        public event PropertyChangedEventHandler? PropertyChanged;
        
        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected bool SetProperty<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return false;
            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }
    }

    public class StatisticalSummary
    {
        public double Mean { get; set; }
        public double Median { get; set; }
        public double StandardDeviation { get; set; }
        public double Variance { get; set; }
        public double Min { get; set; }
        public double Max { get; set; }
        public double Range { get; set; }
        public double Q1 { get; set; }
        public double Q3 { get; set; }
        public double IQR { get; set; }
        public double Skewness { get; set; }
        public double Kurtosis { get; set; }
        public int UniqueValues { get; set; }
        public int MissingValues { get; set; }
        public string MostFrequentValue { get; set; } = string.Empty;
        public int MostFrequentCount { get; set; }
    }

    public class HistogramBin
    {
        public double BinStart { get; set; }
        public double BinEnd { get; set; }
        public double BinCenter { get; set; }
        public int Count { get; set; }
        public string CategoryName { get; set; } = string.Empty;
    }

    public class PreviewInfo
    {
        public int SampleSize { get; set; }
        public int NonNullCount { get; set; }
        public int UniqueValueCount { get; set; }
        public int MissingCount { get; set; }
        public double MissingPercentage => SampleSize > 0 ? (double)MissingCount / SampleSize * 100 : 0;
        public double DataQuality => SampleSize > 0 ? (double)NonNullCount / SampleSize * 100 : 0;
    }
}
