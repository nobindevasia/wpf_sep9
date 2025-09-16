using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using MathNet.Numerics.Statistics;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class CorrelationFeatureSelector : IFeatureSelector
    {
        private readonly MLContext _mlContext;
        private readonly StringBuilder _report;

        public CorrelationFeatureSelector(MLContext mlContext)
        {
            _mlContext = mlContext;
            _report = new StringBuilder();
        }

        private class FeatureVector<T>
        {
            [VectorType]
            public float[] Features { get; set; }
            public T Label { get; set; }
        }

        public async Task<(IDataView transformedData, string[] selectedFeatures, string report)> SelectFeatures(
            MLContext mlContext,
            IDataView data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config)
        {
            _report.Clear();
            _report.AppendLine("\nCorrelation-based Feature Selection Results:");
            _report.AppendLine("----------------------------------------------");

            try
            {
                bool hasFeaturesColumn = data.Schema.GetColumnOrNull("Features").HasValue;
                bool hasIndividualColumns = candidateFeatures.Length > 0 &&
                                           data.Schema.GetColumnOrNull(candidateFeatures[0]).HasValue;

                IDataView preparedData = data;

                var targetColInfo = data.Schema.GetColumnOrNull(targetField);
                if (!targetColInfo.HasValue)
                {
                    throw new InvalidOperationException($"Target column '{targetField}' not found in the dataset");
                }

                bool isTargetNumeric = targetColInfo.Value.Type is NumberDataViewType;

                var targetValues = new List<double>();
                var featureValuesList = new List<float[]>();

                if (modelType == ModelType.Regression)
                {
                    var rows = mlContext.Data.CreateEnumerable<FeatureVector<float>>(
                        PrepareDataForRegression(preparedData, targetField), reuseRowObject: false).ToList();

                    if (!rows.Any() || rows[0].Features == null)
                    {
                        throw new InvalidOperationException("No valid feature data found");
                    }

                    targetValues = rows.Select(r => (double)r.Label).ToList();
                    featureValuesList = rows.Select(r => r.Features).ToList();
                }
                else
                {
                    var rows = mlContext.Data.CreateEnumerable<FeatureVector<long>>(
                        PrepareDataForClassification(preparedData, targetField), reuseRowObject: false).ToList();

                    if (!rows.Any() || rows[0].Features == null)
                    {
                        throw new InvalidOperationException("No valid feature data found");
                    }

                    targetValues = rows.Select(r => (double)r.Label).ToList();
                    featureValuesList = rows.Select(r => r.Features).ToList();
                }

                string[] effectiveFeatureNames;
                if (featureValuesList.Count > 0 && featureValuesList[0].Length > 0)
                {
                    int featureCount = featureValuesList[0].Length;
                    effectiveFeatureNames = Enumerable.Range(0, featureCount)
                        .Select(i => candidateFeatures.Length > i ?
                               candidateFeatures[i] : $"Feature_{i}")
                        .ToArray();
                }
                else
                {
                    throw new InvalidOperationException("No feature values found for correlation analysis");
                }

                var targetCorrelations = new Dictionary<string, double>();

                for (int i = 0; i < effectiveFeatureNames.Length && i < featureValuesList[0].Length; i++)
                {
                    var featureValues = featureValuesList.Select(r => (double)r[i]).ToArray();
                    var correlation = Math.Abs(Correlation.Pearson(featureValues, targetValues.ToArray()));
                    targetCorrelations[effectiveFeatureNames[i]] = correlation;
                }

                var sortedFeatures = targetCorrelations
                    .OrderByDescending(x => x.Value)
                    .ToList();

                _report.AppendLine("\nFeatures Ranked by Target Correlation:");
                foreach (var pair in sortedFeatures)
                {
                    _report.AppendLine($"{pair.Key,-40} | {pair.Value:F4}");
                }

                var selectedFeatures = new List<string>();
                var selectedIndices = new List<int>();

                foreach (var pair in sortedFeatures)
                {
                    if (selectedFeatures.Count >= config.MaxFeatures)
                        break;

                    var currentIndex = Array.IndexOf(effectiveFeatureNames, pair.Key);
                    if (currentIndex == -1 || currentIndex >= featureValuesList[0].Length)
                        continue;

                    var currentValues = featureValuesList.Select(r => (double)r[currentIndex]).ToArray();

                    bool isHighlyCorrelated = false;
                    foreach (var selectedIndex in selectedIndices)
                    {
                        if (selectedIndex >= featureValuesList[0].Length)
                            continue;

                        var selectedValues = featureValuesList.Select(r => (double)r[selectedIndex]).ToArray();
                        var correlation = Math.Abs(Correlation.Pearson(currentValues, selectedValues));

                        if (correlation > config.MulticollinearityThreshold)
                        {
                            isHighlyCorrelated = true;
                            break;
                        }
                    }

                    if (!isHighlyCorrelated)
                    {
                        selectedFeatures.Add(pair.Key);
                        selectedIndices.Add(currentIndex);
                    }
                }

                _report.AppendLine($"\nSelection Summary:");
                _report.AppendLine($"Original features: {effectiveFeatureNames.Length}");
                _report.AppendLine($"Selected features: {selectedFeatures.Count}");
                _report.AppendLine($"Multicollinearity threshold: {config.MulticollinearityThreshold}");
                _report.AppendLine("\nSelected Features:");
                foreach (var feature in selectedFeatures)
                {
                    _report.AppendLine($"- {feature} (correlation with target: {targetCorrelations[feature]:F4})");
                }

                IDataView transformedData;
                if (modelType == ModelType.Regression)
                {
                    var selectedRows = featureValuesList.Zip(targetValues, (features, target) =>
                    new FeatureVector<float>
                    {
                        Features = selectedIndices.Select(i => i < features.Length ? features[i] : 0)
                                                .ToArray(),
                        Label = (float)target
                    }).ToList();

                    transformedData = mlContext.Data.LoadFromEnumerable(selectedRows);
                }
                else
                {
                    var selectedRows = featureValuesList.Zip(targetValues, (features, target) =>
                    new FeatureVector<long>
                    {
                        Features = selectedIndices.Select(i => i < features.Length ? features[i] : 0)
                                                .ToArray(),
                        Label = (long)target
                    }).ToList();

                    transformedData = mlContext.Data.LoadFromEnumerable(selectedRows);
                }

                return (transformedData, selectedFeatures.ToArray(), _report.ToString());
            }
            catch (Exception ex)
            {
                _report.AppendLine($"Error during correlation analysis: {ex.Message}");
                Console.WriteLine($"Full error details: {ex}");
                Console.WriteLine($"Stack Trace: {ex.StackTrace}");
                throw;
            }
        }

        private IDataView PrepareDataForClassification(IDataView data, string targetField)
        {
            IDataView processedData = data;

            if (!data.Schema.GetColumnOrNull("Features").HasValue)
            {
                var featureColumns = data.Schema.Select(col => col.Name)
                    .Where(name => name != targetField)
                    .ToArray();

                var featurePipeline = _mlContext.Transforms.Concatenate("Features", featureColumns);
                processedData = featurePipeline.Fit(data).Transform(data);
            }

            
            var labelColumnInfo = processedData.Schema.GetColumnOrNull("Label");
            if (labelColumnInfo == null)
            {
                
                var labelPipeline = _mlContext.Transforms.CopyColumns("Label", targetField);
                processedData = labelPipeline.Fit(processedData).Transform(processedData);
            }
            else if (labelColumnInfo.Value.Type.RawType == typeof(bool))
            {
                
                var convertPipeline = _mlContext.Transforms.Conversion.ConvertType(
                    "Label", "Label", DataKind.Int64);
                processedData = convertPipeline.Fit(processedData).Transform(processedData);
            }

            return processedData;
        }

        private IDataView PrepareDataForRegression(IDataView data, string targetField)
        {
            IDataView processedData = data;

            if (!data.Schema.GetColumnOrNull("Features").HasValue)
            {
                var featureColumns = data.Schema.Select(col => col.Name)
                    .Where(name => name != targetField)
                    .ToArray();

                var featurePipeline = _mlContext.Transforms.Concatenate("Features", featureColumns);
                processedData = featurePipeline.Fit(data).Transform(data);
            }

            
            var labelColumnInfo = processedData.Schema.GetColumnOrNull("Label");
            if (labelColumnInfo == null)
            {
                
                var labelPipeline = _mlContext.Transforms.CopyColumns("Label", targetField);
                processedData = labelPipeline.Fit(processedData).Transform(processedData);
            }
            else if (labelColumnInfo.Value.Type.RawType == typeof(bool))
            {
                
                var convertPipeline = _mlContext.Transforms.Conversion.ConvertType(
                    "Label", "Label", DataKind.Single);
                processedData = convertPipeline.Fit(processedData).Transform(processedData);
            }

            return processedData;
        }
    }
}