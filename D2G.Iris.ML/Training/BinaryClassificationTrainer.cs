using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.AutoML;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;

namespace D2G.Iris.ML.Training
{
    public class BinaryClassificationTrainer : BaseModelTrainer
    {

        

        public BinaryClassificationTrainer(MLContext mlContext, TrainerFactory trainerFactory)
            : base(mlContext, trainerFactory)
        {
        }


        public override async Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"\nStarting binary classification using {(config.AutoML?.Enabled == true ? "AutoML" : config.TrainingParameters.Algorithm)}");
            try
            {
                
                IDataView labeledData;

                if (dataView.Schema.GetColumnOrNull("Label").HasValue)
                {
                    
                    var labelType = dataView.Schema["Label"].Type;
                    if (labelType.RawType != typeof(bool))
                    {
                        
                        var labelPipeline = mlContext.Transforms.Conversion.ConvertType(
                            outputColumnName: "Label", inputColumnName: "Label", outputKind: DataKind.Boolean);
                        labeledData = labelPipeline.Fit(dataView).Transform(dataView);
                    }
                    else
                    {
                        labeledData = dataView;
                    }
                }
                else
                {
                    
                    var labelPipeline = mlContext.Transforms.CopyColumns(
                            outputColumnName: "RawLabel", inputColumnName: config.TargetField)
                        .Append(mlContext.Transforms.Conversion.ConvertType(
                            outputColumnName: "Label", inputColumnName: "RawLabel", outputKind: DataKind.Boolean));
                    labeledData = labelPipeline.Fit(dataView).Transform(dataView);
                }

                IDataView preparedData = PrepareData(labeledData, featureNames);

                return config.AutoML?.Enabled == true
                    ? await TrainWithAdvancedAutoML(mlContext, preparedData, featureNames, config, processedData)
                    : await TrainWithTraditionalApproach(mlContext, preparedData, featureNames, config, processedData);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in training process: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
                throw;
            }
        }


        private async Task<ITransformer> TrainWithAdvancedAutoML(
            MLContext mlContext,
            IDataView preparedData,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"Maximum experiment time: {config.AutoML.MaxExperimentTimeInSeconds} seconds");
            Console.WriteLine($"Optimizing Metric: {config.AutoML.OptimizingMetric}");

            try
            {

                CreateCacheDirectory();


                if (!Enum.TryParse(config.AutoML.OptimizingMetric, out BinaryClassificationMetric metric))
                {
                    Console.WriteLine($"Warning: Unknown OptimizingMetric '{config.AutoML.OptimizingMetric}', defaulting to {nameof(BinaryClassificationMetric.Accuracy)}");
                    metric = BinaryClassificationMetric.Accuracy;
                }


                var experimentSettings = new BinaryExperimentSettings
                {
                    MaxExperimentTimeInSeconds = (uint)config.AutoML.MaxExperimentTimeInSeconds,
                    OptimizingMetric = metric
                };

                var experiment = mlContext.Auto()
                    .CreateBinaryClassificationExperiment(experimentSettings.MaxExperimentTimeInSeconds);

                var experimentStartTime = DateTime.Now;
                var experimentResult = experiment.Execute(
                    trainData: preparedData,
                    labelColumnName: "Label");
                var experimentDuration = DateTime.Now - experimentStartTime;


                Console.WriteLine($"AutoML experiment completed in {experimentDuration.TotalMinutes:F1} minutes");
                Console.WriteLine($"\n=== AutoML Experiment Summary ===");
                Console.WriteLine($"Models evaluated: {experimentResult.RunDetails.Count()}");

                PrintTopModels(experimentResult.RunDetails, metric);


                var bestRun = experimentResult.BestRun;
                var cleanTrainerName = CleanTrainerName(bestRun.TrainerName);
                Console.WriteLine($"\nBest model: {cleanTrainerName}");
                Console.WriteLine($"Training time: {bestRun.RuntimeInSeconds:F1} seconds");

                double bestMetricValue = GetMetricValue(bestRun.ValidationMetrics, metric);
                Console.WriteLine($"Best {metric} value: {bestMetricValue:F4}");


                PrintBinaryClassificationMetrics(bestRun.ValidationMetrics, cleanTrainerName);

                Console.WriteLine($"Confusion Matrix:\n{bestRun.ValidationMetrics.ConfusionMatrix.GetFormattedConfusionTable()}");


                await SaveModelInfo(
                    bestRun.ValidationMetrics,
                    preparedData,
                    featureNames,
                    config,
                    processedData);


                var safeName = SanitizeFileName(cleanTrainerName);
                var modelPath = $"BinaryClassification_AutoML_{safeName}_Model.zip";
                mlContext.Model.Save(bestRun.Model, preparedData.Schema, modelPath);
                Console.WriteLine($"\nModel saved to: {modelPath}");

                return bestRun.Model;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in AutoML process: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }

                Console.WriteLine("Falling back to traditional approach...");
                return await TrainWithTraditionalApproach(mlContext, preparedData, featureNames, config, processedData);
            }
        }

        private void PrintTopModels(IEnumerable<RunDetail<BinaryClassificationMetrics>> runs, BinaryClassificationMetric metric)
        {
            Console.WriteLine($"\nTop 5 models evaluated (ranked by {metric}):");
            Console.WriteLine("Rank | Model Type                | AUC      | Accuracy | F1 Score | Runtime");
            Console.WriteLine("-----|---------------------------|----------|----------|----------|--------");

            int rank = 1;
            var orderedRuns = OrderRunsByMetric(runs, metric);

            foreach (var run in orderedRuns.Take(5))
            {
                var cleanName = CleanTrainerName(run.TrainerName);
                Console.WriteLine($"{rank,4} | {cleanName,-24} | {run.ValidationMetrics.AreaUnderRocCurve,8:F4} | " +
                                  $"{run.ValidationMetrics.Accuracy,8:F4} | {run.ValidationMetrics.F1Score,8:F4} | " +
                                  $"{run.RuntimeInSeconds,6:F1}s");
                rank++;
            }
        }

        private IEnumerable<RunDetail<BinaryClassificationMetrics>> OrderRunsByMetric(
            IEnumerable<RunDetail<BinaryClassificationMetrics>> runs,
            BinaryClassificationMetric metric)
        {
            return metric switch
            {
                BinaryClassificationMetric.Accuracy => runs.OrderByDescending(r => r.ValidationMetrics.Accuracy),
                BinaryClassificationMetric.AreaUnderRocCurve => runs.OrderByDescending(r => r.ValidationMetrics.AreaUnderRocCurve),
                BinaryClassificationMetric.AreaUnderPrecisionRecallCurve => runs.OrderByDescending(r => r.ValidationMetrics.AreaUnderPrecisionRecallCurve),
                BinaryClassificationMetric.F1Score => runs.OrderByDescending(r => r.ValidationMetrics.F1Score),
                BinaryClassificationMetric.NegativePrecision => runs.OrderByDescending(r => r.ValidationMetrics.NegativePrecision),
                BinaryClassificationMetric.NegativeRecall => runs.OrderByDescending(r => r.ValidationMetrics.NegativeRecall),
                BinaryClassificationMetric.PositivePrecision => runs.OrderByDescending(r => r.ValidationMetrics.PositivePrecision),
                BinaryClassificationMetric.PositiveRecall => runs.OrderByDescending(r => r.ValidationMetrics.PositiveRecall),
                _ => runs.OrderByDescending(r => r.ValidationMetrics.AreaUnderRocCurve)
            };
        }

        private double GetMetricValue(BinaryClassificationMetrics metrics, BinaryClassificationMetric metric)
        {
            return metric switch
            {
                BinaryClassificationMetric.Accuracy => metrics.Accuracy,
                BinaryClassificationMetric.AreaUnderRocCurve => metrics.AreaUnderRocCurve,
                BinaryClassificationMetric.AreaUnderPrecisionRecallCurve => metrics.AreaUnderPrecisionRecallCurve,
                BinaryClassificationMetric.F1Score => metrics.F1Score,
                BinaryClassificationMetric.NegativePrecision => metrics.NegativePrecision,
                BinaryClassificationMetric.NegativeRecall => metrics.NegativeRecall,
                BinaryClassificationMetric.PositivePrecision => metrics.PositivePrecision,
                BinaryClassificationMetric.PositiveRecall => metrics.PositiveRecall,
                _ => metrics.AreaUnderRocCurve
            };
        }

        private async Task<ITransformer> TrainWithTraditionalApproach(
            MLContext mlContext,
            IDataView preparedData,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"Using traditional approach with {config.TrainingParameters.Algorithm}");

            var split = SplitTrainTestData(
                mlContext,
                preparedData,
                config.TrainingParameters.TestFraction);

            var trainer = _trainerFactory.GetTrainer(
                config.ModelType,
                config.TrainingParameters);

            Console.WriteLine($"Training with algorithm: {config.TrainingParameters.Algorithm}");
            Console.WriteLine("Algorithm parameters:");
            if (config.TrainingParameters.AlgorithmParameters != null)
            {
                foreach (var param in config.TrainingParameters.AlgorithmParameters)
                {
                    Console.WriteLine($"  {param.Key}: {param.Value}");
                }
            }

            var pipeline = GetBasePipeline(mlContext)
                .Append(trainer)
                .Append(mlContext.Transforms.CopyColumns("Probability", "Score"));

            var trainingStartTime = DateTime.Now;
            var model = await TrainModelAsync(pipeline, split.TrainSet);
            var trainingDuration = DateTime.Now - trainingStartTime;
            Console.WriteLine($"Training completed in {trainingDuration.TotalSeconds:F1} seconds");

            var metrics = EvaluateBinaryClassification(
                mlContext,
                model,
                split.TestSet,
                config.TrainingParameters.Algorithm);

            Console.WriteLine($"Confusion Matrix:\n{metrics.ConfusionMatrix.GetFormattedConfusionTable()}");

            await SaveModelInfo(
                metrics,
                preparedData,
                featureNames,
                config,
                processedData);

            SaveModel(
                mlContext,
    model,
    preparedData,
    "BinaryClassification",
    config.TrainingParameters.Algorithm,
    featureNames,
    Core.Enums.ModelType.BinaryClassification);

            return model;
        }
        private class BinaryVector
        {
            [VectorType]
            public float[] Features { get; set; }
            public bool Label { get; set; }
        }

        private IDataView PrepareData(IDataView labeledData, string[] featureNames)
        {
            if (labeledData.Schema.GetColumnOrNull("Features").HasValue)
            {
                var temp = labeledData.GetColumn<VBuffer<float>>("Features")
                    .Zip(labeledData.GetColumn<bool>("Label"), (feat, lbl) => new BinaryVector
                    {
                        Features = feat.GetValues().ToArray(),
                        Label = lbl
                    })
                    .ToList();

                var schemaDef = SchemaDefinition.Create(typeof(BinaryVector));
                schemaDef[nameof(BinaryVector.Features)].ColumnType =
                    new VectorDataViewType(NumberDataViewType.Single, featureNames.Length);

                return _mlContext.Data.LoadFromEnumerable(temp, schemaDef);
            }
            else
            {
                return _mlContext.Transforms.Concatenate("Features", featureNames)
                    .Fit(labeledData)
                    .Transform(labeledData);
            }
        }
    }
}