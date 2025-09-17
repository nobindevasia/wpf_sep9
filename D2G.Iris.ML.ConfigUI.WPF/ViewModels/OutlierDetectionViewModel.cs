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
        private double _winsorLowerPercentile = 5.0;
        private double _winsorUpperPercentile = 95.0;
        private bool _applyWinsorization = false;
        private bool _isApplyingWinsorization = false;
        private string _winsorizationProgress = string.Empty;

        public OutlierDetectionViewModel(IDialogService dialogService)
        {
            _dialogService = dialogService;
            _outlierResults = new ObservableCollection<OutlierDetectionResult>();
            _summaryResults = new ObservableCollection<OutlierSummaryResult>();
            _availableColumns = new ObservableCollection<string>();
            _selectedMethod = OutlierDetectionMethod.ZScore;
            
            DetectOutliersCommand = new AsyncRelayCommand(async _ => await DetectOutliersAsync());
            RemoveOutliersCommand = new AsyncRelayCommand(async _ => await RemoveOutliersAsync(), _ => CanRemoveOutliers());
            ApplyWinsorizationCommand = new RelayCommand(_ => ApplyWinsorizationToData(), _ => CanRemoveOutliers());
            SelectAllColumnsCommand = new RelayCommand(_ => SelectAllColumns(), _ => SummaryResults.Any());
            DeselectAllColumnsCommand = new RelayCommand(_ => DeselectAllColumns(), _ => SummaryResults.Any());
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

        public double WinsorLowerPercentile
        {
            get => _winsorLowerPercentile;
            set 
            {
                var clampedValue = Math.Max(0.1, Math.Min(value, 99.8));
                SetProperty(ref _winsorLowerPercentile, clampedValue);
            }
        }

        public double WinsorUpperPercentile
        {
            get => _winsorUpperPercentile;
            set 
            { 
                var clampedValue = Math.Max(0.2, Math.Min(value, 99.9));
                SetProperty(ref _winsorUpperPercentile, clampedValue);
            }
        }

        public bool ApplyWinsorization
        {
            get => _applyWinsorization;
            set => SetProperty(ref _applyWinsorization, value);
        }

        public bool IsApplyingWinsorization
        {
            get => _isApplyingWinsorization;
            set => SetProperty(ref _isApplyingWinsorization, value);
        }

        public string WinsorizationProgress
        {
            get => _winsorizationProgress;
            set => SetProperty(ref _winsorizationProgress, value);
        }

        public Array OutlierDetectionMethods => Enum.GetValues(typeof(OutlierDetectionMethod));

        #endregion

        #region Commands

        public ICommand DetectOutliersCommand { get; }
        public ICommand RemoveOutliersCommand { get; }
        public ICommand ApplyWinsorizationCommand { get; }
        public ICommand SelectAllColumnsCommand { get; }
        public ICommand DeselectAllColumnsCommand { get; }

        #endregion

        #region Public Methods

        public void SetDataTable(DataTable? dataTable, string? targetColumn = null)
        {
            _dataTable = dataTable;
            _targetColumn = targetColumn;
            
            
            if (!string.IsNullOrEmpty(_targetColumn) && _dataTable != null && !_dataTable.Columns.Contains(_targetColumn))
            {
                _dialogService.ShowErrorDialog($"Target column '{_targetColumn}' not found in dataset. Available columns: {string.Join(", ", _dataTable.Columns.Cast<DataColumn>().Select(c => c.ColumnName))}", "Target Column Error");
                _targetColumn = null; 
            }
            
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

        public void SelectAllColumns()
        {
            foreach (var summary in SummaryResults)
            {
                summary.IsSelectedForRemoval = true;
            }
        }

        public void DeselectAllColumns()
        {
            foreach (var summary in SummaryResults)
            {
                summary.IsSelectedForRemoval = false;
            }
        }
       

        public async void ApplyWinsorizationToData()
        {
            if (_dataTable == null || !OutlierResults.Any())
            {
                _dialogService.ShowErrorDialog("No data or outliers available for winsorization.", "Error");
                return;
            }

            var selectedColumns = SummaryResults
                .Where(s => s.IsSelectedForRemoval)
                .Select(s => s.ColumnName)
                .ToHashSet();

            if (!selectedColumns.Any())
            {
                _dialogService.ShowInfoDialog("Please select at least one column for winsorization.", "No Columns Selected");
                return;
            }

            try
            {
                IsApplyingWinsorization = true;
                WinsorizationProgress = "Applying winsorization to selected columns...";
                await Task.Delay(100);

                var transformedCount = 0;
                var totalColumns = selectedColumns.Count;
                var currentColumn = 0;

                foreach (var columnName in selectedColumns)
                {
                    currentColumn++;
                    WinsorizationProgress = $"Processing column: {columnName} ({currentColumn}/{totalColumns})...";
                    await Task.Delay(50);

                    var column = _dataTable.Columns[columnName];
                    if (column == null || !IsNumericColumn(column)) continue;

                    var values = ExtractNumericValues(column);
                    if (values.Count < 4) continue;

                    var sortedValues = values.Select(v => v.Value).OrderBy(x => x).ToList();
                    var lowerBound = CalculatePercentile(sortedValues, WinsorLowerPercentile);
                    var upperBound = CalculatePercentile(sortedValues, WinsorUpperPercentile);

                    foreach (var value in values)
                    {
                        if (value.Value < lowerBound || value.Value > upperBound)
                        {
                            var winsorizedValue = value.Value < lowerBound ? lowerBound : upperBound;
                            _dataTable.Rows[value.Index][columnName] = winsorizedValue;
                            transformedCount++;
                        }
                    }
                }

                _dialogService.ShowInfoDialog($"Winsorization completed. {transformedCount} values were transformed.", "Winsorization Complete");
                
                OutlierResults.Clear();
                foreach (var summary in SummaryResults.Where(s => selectedColumns.Contains(s.ColumnName)))
                {
                    summary.OutlierCount = 0;
                    summary.OutlierPercentage = 0;
                    summary.Status = "Winsorized";
                    summary.IsSelectedForRemoval = false;
                }
                RemoveOutliersEnabled = true;
            }
            catch (Exception ex)
            {
                _dialogService.ShowErrorDialog($"Error during winsorization: {ex.Message}", "Error");
            }
            finally
            {
                IsApplyingWinsorization = false;
                WinsorizationProgress = string.Empty;
            }
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
            var totalRows = _dataTable!.Rows.Count;
            var values = new List<ValueInfo>(totalRows);

            
            for (int i = 0; i < totalRows; i++)
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
            var outliers = new List<OutlierInfo>(values.Count / 10); 

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
            var outliers = new List<OutlierInfo>(values.Count / 10); 

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
            var outliers = new List<OutlierInfo>(values.Count / 10); 

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

            
            var selectedColumns = SummaryResults
                .Where(s => s.IsSelectedForRemoval)
                .Select(s => s.ColumnName)
                .ToHashSet();

            if (!selectedColumns.Any())
            {
                _dialogService.ShowInfoDialog("Please select at least one column for outlier removal.", "No Columns Selected");
                return;
            }

            
            if (ApplyWinsorization)
            {
                ApplyWinsorizationToData();
                return;
            }

            try
            {
                IsAnalyzing = true;
                AnalysisMessage = "Analyzing selected columns for outlier removal...";

                await Task.Delay(100);

                
                Console.WriteLine("=== SELECTIVE OUTLIER REMOVAL ===");
                Console.WriteLine($"Selected columns: {string.Join(", ", selectedColumns)}");
                Console.WriteLine($"DataTable Columns ({_dataTable.Columns.Count}):");
                foreach (DataColumn col in _dataTable.Columns)
                {
                    Console.WriteLine($"  - {col.ColumnName} ({col.DataType.Name})");
                }

                Console.WriteLine($"Target Column Name: '{_targetColumn}'");
                Console.WriteLine($"Target Column Exists: {!string.IsNullOrEmpty(_targetColumn) && _dataTable.Columns.Contains(_targetColumn)}");

                
                var targetValueCounts = new Dictionary<string, int>();
                int nullCount = 0;

                if (!string.IsNullOrEmpty(_targetColumn))
                {
                    var targetCol = _dataTable.Columns[_targetColumn];
                    if (targetCol != null)
                    {
                        Console.WriteLine($"Target Column Type: {targetCol.DataType.Name}");
                    }

                    
                    foreach (DataRow row in _dataTable.Rows)
                    {
                        var value = row[_targetColumn];
                        if (value == null || value == DBNull.Value)
                        {
                            nullCount++;
                        }
                        else
                        {
                            var stringValue = value.ToString() ?? string.Empty;
                            targetValueCounts[stringValue] = targetValueCounts.ContainsKey(stringValue)
                                ? targetValueCounts[stringValue] + 1 : 1;
                        }
                    }

                    Console.WriteLine($"Target column distribution (before removal):");
                    Console.WriteLine($"  NULL/DBNull values: {nullCount}");
                    foreach (var kvp in targetValueCounts.OrderByDescending(x => x.Value))
                    {
                        Console.WriteLine($"  '{kvp.Key}': {kvp.Value:N0} samples");
                    }
                }

                
                var selectedOutliers = OutlierResults
                    .Where(r => selectedColumns.Contains(r.ColumnName))
                    .ToList();

                var rowIndicesToRemove = selectedOutliers
                    .Select(r => r.RowIndex)
                    .Distinct()
                    .OrderBy(i => i)
                    .ToList();

                Console.WriteLine($"Selected columns outliers: {selectedOutliers.Count:N0}");
                Console.WriteLine($"Total rows to remove: {rowIndicesToRemove.Count:N0}");
                Console.WriteLine($"Original dataset size: {_dataTable.Rows.Count:N0}");
                Console.WriteLine($"Percentage to remove: {(double)rowIndicesToRemove.Count / _dataTable.Rows.Count * 100:F2}%");

                
                var removedTargetValues = new Dictionary<string, int>();
                int removedNulls = 0;

                if (!string.IsNullOrEmpty(_targetColumn))
                {
                    foreach (var rowIndex in rowIndicesToRemove.Take(100))
                    {
                        if (rowIndex >= 0 && rowIndex < _dataTable.Rows.Count)
                        {
                            var value = _dataTable.Rows[rowIndex][_targetColumn];
                            if (value == null || value == DBNull.Value)
                            {
                                removedNulls++;
                            }
                            else
                            {
                                var stringValue = value.ToString() ?? string.Empty;
                                removedTargetValues[stringValue] = removedTargetValues.ContainsKey(stringValue)
                                    ? removedTargetValues[stringValue] + 1 : 1;
                            }
                        }
                    }
                }

                Console.WriteLine($"Target values being removed (first 100 outliers):");
                Console.WriteLine($"  NULL values: {removedNulls}");
                foreach (var kvp in removedTargetValues)
                {
                    Console.WriteLine($"  '{kvp.Key}': {kvp.Value} samples");
                }

                
                var warningMessage = $"SELECTIVE OUTLIER REMOVAL:\n\n" +
                                   $"Selected columns: {string.Join(", ", selectedColumns)}\n\n" +
                                   $"Dataset size: {_dataTable.Rows.Count:N0} rows\n" +
                                   $"Outliers to remove: {rowIndicesToRemove.Count:N0} rows ({(double)rowIndicesToRemove.Count / _dataTable.Rows.Count * 100:F2}%)\n\n";

                if (!string.IsNullOrEmpty(_targetColumn))
                {
                    warningMessage += $"Target column: '{_targetColumn}'\n" +
                                    $"Current class distribution:\n" +
                                    string.Join("\n", targetValueCounts.Select(kvp => $"  {kvp.Key}: {kvp.Value:N0} samples")) +
                                    (nullCount > 0 ? $"\n  NULL: {nullCount:N0} samples" : "") +
                                    "\n\n";
                }

                warningMessage += $"Do you want to proceed with selective outlier removal?\n\n" +
                                $"⚠️ Large percentage removal may cause class imbalance issues!";

                if (!_dialogService.ShowConfirmationDialog(warningMessage, "Confirm Selective Outlier Removal"))
                {
                    IsAnalyzing = false;
                    AnalysisMessage = string.Empty;
                    return;
                }

                AnalysisMessage = "Removing outliers from selected columns...";

                
                var originalRowCount = _dataTable.Rows.Count;

                foreach (var rowIndex in rowIndicesToRemove.OrderByDescending(i => i))
                {
                    if (rowIndex >= 0 && rowIndex < _dataTable.Rows.Count)
                    {
                        _dataTable.Rows.RemoveAt(rowIndex);
                    }
                }

                var newRowCount = _dataTable.Rows.Count;
                var removedCount = originalRowCount - newRowCount;

                
                var finalTargetValueCounts = new Dictionary<string, int>();
                int finalNullCount = 0;

                if (!string.IsNullOrEmpty(_targetColumn))
                {
                    foreach (DataRow row in _dataTable.Rows)
                    {
                        var value = row[_targetColumn];
                        if (value == null || value == DBNull.Value)
                        {
                            finalNullCount++;
                        }
                        else
                        {
                            var stringValue = value.ToString() ?? string.Empty;
                            finalTargetValueCounts[stringValue] = finalTargetValueCounts.ContainsKey(stringValue)
                                ? finalTargetValueCounts[stringValue] + 1 : 1;
                        }
                    }
                }

                Console.WriteLine($"Final target column distribution (after removal):");
                Console.WriteLine($"  NULL/DBNull values: {finalNullCount}");
                foreach (var kvp in finalTargetValueCounts.OrderByDescending(x => x.Value))
                {
                    Console.WriteLine($"  '{kvp.Key}': {kvp.Value:N0} samples");
                }

                
                OutlierResults.Clear();
                foreach (var summary in SummaryResults.Where(s => selectedColumns.Contains(s.ColumnName)))
                {
                    summary.OutlierCount = 0;
                    summary.OutlierPercentage = 0;
                    summary.Status = "Cleaned";
                    summary.IsSelectedForRemoval = false;
                }
                RemoveOutliersEnabled = true;

                
                string resultMessage = $"Selective outlier removal completed:\n\n" +
                                      $"Selected columns: {string.Join(", ", selectedColumns)}\n" +
                                      $"Original dataset: {originalRowCount:N0} rows\n" +
                                      $"Cleaned dataset: {newRowCount:N0} rows\n" +
                                      $"Removed: {removedCount:N0} rows\n\n";

                if (!string.IsNullOrEmpty(_targetColumn))
                {
                    resultMessage += $"Final Label Distribution:\n";

                    if (finalTargetValueCounts.Any())
                    {
                        resultMessage += string.Join("\n", finalTargetValueCounts.Select(kvp => $"  {kvp.Key}: {kvp.Value:N0} samples"));
                    }
                    else
                    {
                        resultMessage += "  ⚠️ NO CLASS SAMPLES REMAINING!";
                    }

                    if (finalNullCount > 0)
                    {
                        resultMessage += $"\n  NULL: {finalNullCount:N0} samples";
                    }

                    
                    bool hasClass0 = finalTargetValueCounts.ContainsKey("0") || finalTargetValueCounts.ContainsKey("False");
                    bool hasClass1 = finalTargetValueCounts.ContainsKey("1") || finalTargetValueCounts.ContainsKey("True");

                    if (!hasClass0 || !hasClass1)
                    {
                        resultMessage += "\n\n⚠️ WARNING: Missing class data may cause training errors!";
                    }

                    resultMessage += "\n\n";
                }

                resultMessage += "The cleaned dataset will be used for training.";

                _dialogService.ShowInfoDialog(resultMessage, "Selective Outliers Removed");
            }
            catch (Exception ex)
            {
                _dialogService.ShowErrorDialog($"Error removing outliers: {ex.Message}", "Error");
                Console.WriteLine($"Outlier removal error: {ex}");
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

    public class OutlierSummaryResult : INotifyPropertyChanged
    {
        private bool _isSelectedForRemoval = true;

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
        
        public bool IsSelectedForRemoval
        {
            get => _isSelectedForRemoval;
            set
            {
                _isSelectedForRemoval = value;
                OnPropertyChanged();
            }
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
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