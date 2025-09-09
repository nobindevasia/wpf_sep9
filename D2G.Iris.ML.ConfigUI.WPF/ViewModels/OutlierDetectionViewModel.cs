using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Data;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Input;
using D2G.Iris.ML.ConfigUI.WPF.Commands;
using D2G.Iris.ML.ConfigUI.WPF.Services;

namespace D2G.Iris.ML.ConfigUI.WPF.ViewModels
{
    public class OutlierDetectionViewModel : INotifyPropertyChanged
    {
        private readonly IDialogService _dialogService;
        private DataTable? _dataTable;
        private string? _targetColumn;
        private ObservableCollection<OutlierDetectionResult> _outlierResults;
        private ObservableCollection<OutlierSummaryResult> _summaryResults;
        private OutlierDetectionMethod _selectedMethod;
        private bool _isAnalyzing;
        private string _analysisMessage = string.Empty;
        private ObservableCollection<string> _availableColumns;
        private bool _removeOutliersEnabled = false;
        private double _zScoreThreshold = 3.0;
        private double _iqrMultiplier = 1.5;
        private double _modifiedZScoreThreshold = 3.5;

        public OutlierDetectionViewModel(IDialogService dialogService)
        {
            _dialogService = dialogService;
            _outlierResults = new ObservableCollection<OutlierDetectionResult>();
            _summaryResults = new ObservableCollection<OutlierSummaryResult>();
            _availableColumns = new ObservableCollection<string>();
            _selectedMethod = OutlierDetectionMethod.ZScore;
            
            DetectOutliersCommand = new AsyncRelayCommand(async _ => await DetectOutliersAsync());
            RemoveOutliersCommand = new AsyncRelayCommand(async _ => await RemoveOutliersAsync(), _ => CanRemoveOutliers());
        }

        #region Properties

        public ObservableCollection<OutlierDetectionResult> OutlierResults
        {
            get => _outlierResults;
            set => SetProperty(ref _outlierResults, value);
        }

        public ObservableCollection<OutlierSummaryResult> SummaryResults
        {
            get => _summaryResults;
            set => SetProperty(ref _summaryResults, value);
        }

        public ObservableCollection<string> AvailableColumns
        {
            get => _availableColumns;
            set => SetProperty(ref _availableColumns, value);
        }

        public OutlierDetectionMethod SelectedMethod
        {
            get => _selectedMethod;
            set => SetProperty(ref _selectedMethod, value);
        }

        public bool IsAnalyzing
        {
            get => _isAnalyzing;
            set => SetProperty(ref _isAnalyzing, value);
        }

        public string AnalysisMessage
        {
            get => _analysisMessage;
            set => SetProperty(ref _analysisMessage, value);
        }

        public double ZScoreThreshold
        {
            get => _zScoreThreshold;
            set => SetProperty(ref _zScoreThreshold, value);
        }

        public double IQRMultiplier
        {
            get => _iqrMultiplier;
            set => SetProperty(ref _iqrMultiplier, value);
        }

        public double ModifiedZScoreThreshold
        {
            get => _modifiedZScoreThreshold;
            set => SetProperty(ref _modifiedZScoreThreshold, value);
        }

        public bool RemoveOutliersEnabled
        {
            get => _removeOutliersEnabled;
            set => SetProperty(ref _removeOutliersEnabled, value);
        }

        public Array OutlierDetectionMethods => Enum.GetValues(typeof(OutlierDetectionMethod));

        #endregion

        #region Commands

        public ICommand DetectOutliersCommand { get; }
        public ICommand RemoveOutliersCommand { get; }

        #endregion

        #region Public Methods

        public void SetDataTable(DataTable? dataTable, string? targetColumn = null)
        {
            _dataTable = dataTable;
            _targetColumn = targetColumn;
            UpdateAvailableColumns();
        }

        public DataTable? GetCleanedDataTable()
        {
            return _dataTable;
        }

        public bool HasOutliersBeenRemoved()
        {
            return RemoveOutliersEnabled;
        }

        #endregion

        #region Private Methods

        private void UpdateAvailableColumns()
        {
            AvailableColumns.Clear();
            
            if (_dataTable == null) return;

            var numericColumns = _dataTable.Columns.Cast<DataColumn>()
                .Where(c => IsNumericColumn(c) && !IsTargetColumn(c.ColumnName))
                .Select(c => c.ColumnName)
                .OrderBy(name => name);

            foreach (var columnName in numericColumns)
            {
                AvailableColumns.Add(columnName);
            }
        }

        private bool IsTargetColumn(string columnName)
        {
            return !string.IsNullOrEmpty(_targetColumn) && 
                   columnName.Equals(_targetColumn, StringComparison.OrdinalIgnoreCase);
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

        private async Task DetectOutliersAsync()
        {
            if (_dataTable == null)
            {
                _dialogService.ShowErrorDialog("No data available for outlier detection.", "Error");
                return;
            }

            try
            {
                IsAnalyzing = true;
                AnalysisMessage = "Detecting outliers across all numeric columns...";
                OutlierResults.Clear();
                SummaryResults.Clear();

                await Task.Delay(100);

                var numericColumns = _dataTable.Columns.Cast<DataColumn>()
                    .Where(c => IsNumericColumn(c) && !IsTargetColumn(c.ColumnName))
                    .ToList();

                if (!numericColumns.Any())
                {
                    _dialogService.ShowInfoDialog("No numeric columns found for outlier analysis.", "Information");
                    return;
                }

                // Debug: Log which columns are being analyzed
                var columnNames = string.Join(", ", numericColumns.Select(c => c.ColumnName));
                Console.WriteLine($"Analyzing columns for outliers: {columnNames}");
                if (!string.IsNullOrEmpty(_targetColumn))
                {
                    Console.WriteLine($"Target column '{_targetColumn}' excluded from outlier analysis.");
                }

                AnalysisMessage = $"Analyzing {numericColumns.Count} numeric columns using {SelectedMethod}...";
                await Task.Delay(100);

                var totalOutliers = 0;
                var totalValues = 0;

                foreach (var column in numericColumns)
                {
                    AnalysisMessage = $"Processing column: {column.ColumnName}...";
                    await Task.Delay(50);

                    var values = ExtractNumericValues(column);
                    if (!values.Any()) continue;

                    var outliers = SelectedMethod switch
                    {
                        OutlierDetectionMethod.ZScore => DetectZScoreOutliers(values),
                        OutlierDetectionMethod.IQR => DetectIQROutliers(values),
                        OutlierDetectionMethod.ModifiedZScore => DetectModifiedZScoreOutliers(values),
                        _ => new List<OutlierInfo>()
                    };

                    // Add detailed results
                    foreach (var outlier in outliers.OrderByDescending(o => Math.Abs(o.Score)))
                    {
                        OutlierResults.Add(new OutlierDetectionResult
                        {
                            RowIndex = outlier.Index,
                            Value = outlier.Value,
                            Score = outlier.Score,
                            Method = SelectedMethod.ToString(),
                            ColumnName = column.ColumnName,
                            Severity = CalculateSeverity(outlier.Score)
                        });
                    }

                    // Add summary result
                    var outlierCount = outliers.Count;
                    var valueCount = values.Count;
                    var percentage = valueCount > 0 ? (double)outlierCount / valueCount * 100 : 0;
                    
                    var severityCounts = outliers.GroupBy(o => CalculateSeverity(o.Score))
                        .ToDictionary(g => g.Key, g => g.Count());

                    SummaryResults.Add(new OutlierSummaryResult
                    {
                        ColumnName = column.ColumnName,
                        TotalValues = valueCount,
                        OutlierCount = outlierCount,
                        OutlierPercentage = percentage,
                        Method = SelectedMethod.ToString(),
                        LowSeverityCount = severityCounts.GetValueOrDefault("Low", 0),
                        MediumSeverityCount = severityCounts.GetValueOrDefault("Medium", 0),
                        HighSeverityCount = severityCounts.GetValueOrDefault("High", 0),
                        ExtremeSeverityCount = severityCounts.GetValueOrDefault("Extreme", 0),
                        MaxScore = outliers.Any() ? outliers.Max(o => o.Score) : 0,
                        Status = GetColumnStatus(percentage)
                    });

                    totalOutliers += outlierCount;
                    totalValues += valueCount;
                }

                var overallPercentage = totalValues > 0 ? (double)totalOutliers / totalValues * 100 : 0;

                _dialogService.ShowInfoDialog(
                    $"Outlier detection completed for {numericColumns.Count} columns.\n\n" +
                    $"Found {totalOutliers} outliers out of {totalValues} total values ({overallPercentage:F2}%).\n\n" +
                    $"Check the Summary tab for detailed column-wise results.",
                    "Analysis Complete");
            }
            catch (Exception ex)
            {
                _dialogService.ShowErrorDialog($"Error detecting outliers: {ex.Message}", "Analysis Error");
            }
            finally
            {
                IsAnalyzing = false;
                AnalysisMessage = string.Empty;
            }
        }

        private List<ValueInfo> ExtractNumericValues(DataColumn column)
        {
            var values = new List<ValueInfo>();
            
            for (int i = 0; i < _dataTable!.Rows.Count; i++)
            {
                var value = _dataTable.Rows[i][column];
                if (value != null && value != DBNull.Value)
                {
                    if (double.TryParse(value.ToString(), out double numericValue))
                    {
                        values.Add(new ValueInfo { Index = i, Value = numericValue });
                    }
                }
            }

            return values;
        }

        private List<OutlierInfo> DetectZScoreOutliers(List<ValueInfo> values)
        {
            var outliers = new List<OutlierInfo>();
            
            if (values.Count < 2) return outliers;

            var mean = values.Average(v => v.Value);
            var variance = values.Sum(v => Math.Pow(v.Value - mean, 2)) / values.Count;
            var standardDeviation = Math.Sqrt(variance);

            if (standardDeviation == 0) return outliers;

            foreach (var value in values)
            {
                var zScore = Math.Abs(value.Value - mean) / standardDeviation;
                if (zScore > ZScoreThreshold)
                {
                    outliers.Add(new OutlierInfo
                    {
                        Index = value.Index,
                        Value = value.Value,
                        Score = zScore
                    });
                }
            }

            return outliers;
        }

        private List<OutlierInfo> DetectIQROutliers(List<ValueInfo> values)
        {
            var outliers = new List<OutlierInfo>();
            
            if (values.Count < 4) return outliers;

            var sortedValues = values.OrderBy(v => v.Value).ToList();
            var q1 = CalculatePercentile(sortedValues.Select(v => v.Value).ToList(), 25);
            var q3 = CalculatePercentile(sortedValues.Select(v => v.Value).ToList(), 75);
            var iqr = q3 - q1;

            if (iqr == 0) return outliers;

            var lowerBound = q1 - (IQRMultiplier * iqr);
            var upperBound = q3 + (IQRMultiplier * iqr);

            foreach (var value in values)
            {
                if (value.Value < lowerBound || value.Value > upperBound)
                {
                    var score = value.Value < lowerBound 
                        ? (lowerBound - value.Value) / iqr
                        : (value.Value - upperBound) / iqr;
                    
                    outliers.Add(new OutlierInfo
                    {
                        Index = value.Index,
                        Value = value.Value,
                        Score = score
                    });
                }
            }

            return outliers;
        }

        private List<OutlierInfo> DetectModifiedZScoreOutliers(List<ValueInfo> values)
        {
            var outliers = new List<OutlierInfo>();
            
            if (values.Count < 2) return outliers;

            var median = CalculateMedian(values.Select(v => v.Value).ToList());
            var deviations = values.Select(v => Math.Abs(v.Value - median)).ToList();
            var mad = CalculateMedian(deviations);

            if (mad == 0) return outliers;

            var threshold = ModifiedZScoreThreshold;
            
            foreach (var value in values)
            {
                var modifiedZScore = 0.6745 * (value.Value - median) / mad;
                if (Math.Abs(modifiedZScore) > threshold)
                {
                    outliers.Add(new OutlierInfo
                    {
                        Index = value.Index,
                        Value = value.Value,
                        Score = Math.Abs(modifiedZScore)
                    });
                }
            }

            return outliers;
        }

        private double CalculatePercentile(List<double> sortedValues, double percentile)
        {
            if (!sortedValues.Any()) return 0;
            
            var n = sortedValues.Count;
            if (n == 1) return sortedValues[0];
            
            var index = percentile / 100.0 * (n - 1);
            
            if (index <= 0) return sortedValues[0];
            if (index >= n - 1) return sortedValues[n - 1];
            
            if (index == Math.Floor(index))
            {
                return sortedValues[(int)index];
            }
            
            var lower = (int)Math.Floor(index);
            var upper = (int)Math.Ceiling(index);
            var weight = index - lower;
            
            return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
        }

        private double CalculateMedian(List<double> values)
        {
            if (!values.Any()) return 0;
            
            var sorted = values.OrderBy(x => x).ToList();
            var n = sorted.Count;
            
            return n % 2 == 0
                ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                : sorted[n / 2];
        }

        private string CalculateSeverity(double score)
        {
            return score switch
            {
                < 2 => "Low",
                < 3 => "Medium",
                < 5 => "High",
                _ => "Extreme"
            };
        }

        private string GetColumnStatus(double outlierPercentage)
        {
            return outlierPercentage switch
            {
                <= 1.0 => "Good",
                <= 5.0 => "Fair",
                <= 10.0 => "Poor",
                _ => "Critical"
            };
        }

        private bool CanRemoveOutliers()
        {
            return OutlierResults.Any() && !IsAnalyzing;
        }

        private async Task RemoveOutliersAsync()
        {
            if (_dataTable == null || !OutlierResults.Any())
            {
                _dialogService.ShowErrorDialog("No outliers to remove.", "Error");
                return;
            }

            try
            {
                IsAnalyzing = true;
                AnalysisMessage = "Removing outliers from dataset...";

                await Task.Delay(100);

                // Get unique row indices of outliers to remove
                var rowIndicesToRemove = OutlierResults
                    .Select(r => r.RowIndex)
                    .Distinct()
                    .OrderByDescending(i => i) // Remove from end to start to maintain indices
                    .ToList();

                var originalRowCount = _dataTable.Rows.Count;

                // Remove rows from DataTable
                foreach (var rowIndex in rowIndicesToRemove)
                {
                    if (rowIndex >= 0 && rowIndex < _dataTable.Rows.Count)
                    {
                        _dataTable.Rows.RemoveAt(rowIndex);
                    }
                }

                var newRowCount = _dataTable.Rows.Count;
                var removedCount = originalRowCount - newRowCount;

                // Clear results since they're no longer valid
                OutlierResults.Clear();
                SummaryResults.Clear();

                // Update the flag
                RemoveOutliersEnabled = true;

                _dialogService.ShowInfoDialog(
                    $"Successfully removed {removedCount} rows containing outliers.\n\n" +
                    $"Original dataset: {originalRowCount} rows\n" +
                    $"Cleaned dataset: {newRowCount} rows\n\n" +
                    $"The cleaned dataset will be used for training.",
                    "Outliers Removed");
            }
            catch (Exception ex)
            {
                _dialogService.ShowErrorDialog($"Error removing outliers: {ex.Message}", "Error");
            }
            finally
            {
                IsAnalyzing = false;
                AnalysisMessage = string.Empty;
            }
        }


        #endregion

        #region INotifyPropertyChanged

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

        #endregion
    }

    #region Supporting Classes

    public enum OutlierDetectionMethod
    {
        ZScore,
        IQR,
        ModifiedZScore
    }

    public class OutlierDetectionResult
    {
        public int RowIndex { get; set; }
        public double Value { get; set; }
        public double Score { get; set; }
        public string Method { get; set; } = string.Empty;
        public string ColumnName { get; set; } = string.Empty;
        public string Severity { get; set; } = string.Empty;
    }

    public class OutlierSummaryResult
    {
        public string ColumnName { get; set; } = string.Empty;
        public int TotalValues { get; set; }
        public int OutlierCount { get; set; }
        public double OutlierPercentage { get; set; }
        public string Method { get; set; } = string.Empty;
        public int LowSeverityCount { get; set; }
        public int MediumSeverityCount { get; set; }
        public int HighSeverityCount { get; set; }
        public int ExtremeSeverityCount { get; set; }
        public double MaxScore { get; set; }
        public string Status { get; set; } = string.Empty;
    }

    public class OutlierInfo
    {
        public int Index { get; set; }
        public double Value { get; set; }
        public double Score { get; set; }
    }

    public class ValueInfo
    {
        public int Index { get; set; }
        public double Value { get; set; }
    }

    #endregion
}