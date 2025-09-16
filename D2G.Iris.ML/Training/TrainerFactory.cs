using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;

namespace D2G.Iris.ML.Training
{
    public class TrainerFactory
    {
        private readonly MLContext _mlContext;

        public TrainerFactory(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        public IEstimator<ITransformer> GetTrainer(ModelType modelType, TrainingParameters parameters)
        {
            return modelType switch
            {
                ModelType.BinaryClassification => GetBinaryClassificationTrainer(parameters.Algorithm, parameters.AlgorithmParameters),
                ModelType.MultiClassClassification => GetMultiClassClassificationTrainer(parameters.Algorithm, parameters.AlgorithmParameters),
                ModelType.Regression => GetRegressionTrainer(parameters.Algorithm, parameters.AlgorithmParameters),
                _ => throw new ArgumentException($"Unsupported model type: {modelType}")
            };
        }

        private IEstimator<ITransformer> GetBinaryClassificationTrainer(string algorithm, Dictionary<string, object> parameters)
        {
            var optionsType = AlgorithmRegistry.GetOptionsType(algorithm, ModelType.BinaryClassification);
            if (optionsType == null)
                throw new ArgumentException($"Unsupported binary classification algorithm: {algorithm}");

            return algorithm.ToLower() switch
            {
                "fastforest" => CreateTrainer(_mlContext.BinaryClassification.Trainers.FastForest,
                    (FastForestBinaryTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "fasttree" => CreateTrainer(_mlContext.BinaryClassification.Trainers.FastTree,
                    (FastTreeBinaryTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "lightgbm" => CreateTrainer(_mlContext.BinaryClassification.Trainers.LightGbm,
                    (LightGbmBinaryTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "sdcalogisticregression" => CreateTrainer(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression,
                    (SdcaLogisticRegressionBinaryTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "averagedperceptron" => CreateTrainer(_mlContext.BinaryClassification.Trainers.AveragedPerceptron,
                    (AveragedPerceptronTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "linearsvm" => CreateTrainer(_mlContext.BinaryClassification.Trainers.LinearSvm,
                    (LinearSvmTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "ldsvm" => CreateTrainer(_mlContext.BinaryClassification.Trainers.LdSvm,
                    (LdSvmTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "sdca" => CreateTrainer(_mlContext.BinaryClassification.Trainers.SdcaNonCalibrated,
                    (SdcaNonCalibratedBinaryTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "sgdcalibrated" => CreateTrainer(_mlContext.BinaryClassification.Trainers.SgdCalibrated,
                    (SgdCalibratedTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "symbolicsgdlogisticregression" => CreateTrainer(_mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression,
                    (SymbolicSgdLogisticRegressionBinaryTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "gam" => CreateTrainer(_mlContext.BinaryClassification.Trainers.Gam,
                    (GamBinaryTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "fieldawarefactorizationmachine" => CreateTrainer(_mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine,
                    (FieldAwareFactorizationMachineTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "lbfgslogisticregression" => CreateTrainer(_mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression,
                    (LbfgsLogisticRegressionBinaryTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                _ => throw new ArgumentException($"Unsupported binary classification algorithm: {algorithm}")
            };
        }

        private IEstimator<ITransformer> GetMultiClassClassificationTrainer(string algorithm, Dictionary<string, object> parameters)
        {
            var optionsType = AlgorithmRegistry.GetOptionsType(algorithm, ModelType.MultiClassClassification);
            if (optionsType == null)
                throw new ArgumentException($"Unsupported multiclass classification algorithm: {algorithm}");

            return algorithm.ToLower() switch
            {
                "lightgbm" => CreateTrainer(_mlContext.MulticlassClassification.Trainers.LightGbm,
                    (LightGbmMulticlassTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "sdcamaximumentropy" => CreateTrainer(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy,
                    (SdcaMaximumEntropyMulticlassTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "sdca" => CreateTrainer(_mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated,
                    (SdcaNonCalibratedMulticlassTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "fasttree" => _mlContext.MulticlassClassification.Trainers.OneVersusAll(
                            _mlContext.BinaryClassification.Trainers.FastTree(
                            new FastTreeBinaryTrainer.Options()
                            {
                                LabelColumnName = "Label",
                                FeatureColumnName = "Features"
                            })),

                "fastforest" => _mlContext.MulticlassClassification.Trainers.OneVersusAll(
                            _mlContext.BinaryClassification.Trainers.FastForest(
                            new FastForestBinaryTrainer.Options()
                            {
                                LabelColumnName = "Label",
                                FeatureColumnName = "Features"
                            })),

                "lbfgsmaximumentropy" => CreateTrainer(_mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy,
                    (LbfgsMaximumEntropyMulticlassTrainer.Options)Activator.CreateInstance(optionsType), parameters),
                _ => throw new ArgumentException($"Unsupported multiclass classification algorithm: {algorithm}")
            };
        }

        private IEstimator<ITransformer> GetRegressionTrainer(string algorithm, Dictionary<string, object> parameters)
        {
            var optionsType = AlgorithmRegistry.GetOptionsType(algorithm, ModelType.Regression);
            if (optionsType == null)
                throw new ArgumentException($"Unsupported regression algorithm: {algorithm}");

            return algorithm.ToLower() switch
            {
                "fastforest" => CreateTrainer(_mlContext.Regression.Trainers.FastForest,
                    (FastForestRegressionTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "fasttree" => CreateTrainer(_mlContext.Regression.Trainers.FastTree,
                    (FastTreeRegressionTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "lightgbm" => CreateTrainer(_mlContext.Regression.Trainers.LightGbm,
                    (LightGbmRegressionTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "ols" => CreateTrainer(_mlContext.Regression.Trainers.Ols,
                    (OlsTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "onlinegradientdescent" => CreateTrainer(_mlContext.Regression.Trainers.OnlineGradientDescent,
                    (OnlineGradientDescentTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "gam" => CreateTrainer(_mlContext.Regression.Trainers.Gam,
                    (GamRegressionTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "sdca" => CreateTrainer(_mlContext.Regression.Trainers.Sdca,
                    (SdcaRegressionTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "fasttreetweedie" => CreateTrainer(_mlContext.Regression.Trainers.FastTreeTweedie,
                    (FastTreeTweedieTrainer.Options)Activator.CreateInstance(optionsType), parameters),

                "lbfgspoissonregression" => CreateTrainer(_mlContext.Regression.Trainers.LbfgsPoissonRegression,
                    (LbfgsPoissonRegressionTrainer.Options)Activator.CreateInstance(optionsType), parameters),
                _ => throw new ArgumentException($"Unsupported regression algorithm: {algorithm}")
            };
        }

        private IEstimator<ITransformer> CreateTrainer<TOptions>(
            Func<TOptions, IEstimator<ITransformer>> trainerBuilder,
            TOptions options,
            Dictionary<string, object> parameters) where TOptions : class
        {
            options.GetType().GetProperty("LabelColumnName")?.SetValue(options, "Label");
            options.GetType().GetProperty("FeatureColumnName")?.SetValue(options, "Features");
            ApplyParameters(options, parameters);
            return trainerBuilder(options);
        }

        private static void ApplyParameters<T>(T options, Dictionary<string, object> parameters)
        {
            if (parameters == null) return;

            var type = typeof(T);
            var properties = ParameterHelper.GetConfigurableProperties(type);
            var fields = ParameterHelper.GetConfigurableFields(type);

            var allMembers = new List<MemberInfo>();
            allMembers.AddRange(properties.Cast<MemberInfo>());
            allMembers.AddRange(fields.Cast<MemberInfo>());

            var memberDict = allMembers.ToDictionary(m => m.Name.ToLower(), m => m, StringComparer.OrdinalIgnoreCase);

            foreach (var (key, value) in parameters)
            {
                var cleanKey = key.Split('(')[0].Trim();
                var lookupKey = cleanKey.ToLower();

                if (!memberDict.TryGetValue(lookupKey, out var member))
                {
                    Console.WriteLine($"Warning: Parameter '{cleanKey}' is not recognized on {type.Name}.");
                    Console.WriteLine($"Available parameters: {string.Join(", ", memberDict.Keys)}");
                    continue;
                }

                try
                {
                    var targetType = member switch
                    {
                        PropertyInfo prop => prop.PropertyType,
                        FieldInfo field => field.FieldType,
                        _ => throw new InvalidOperationException("Unexpected member type")
                    };

                    var convertedValue = value switch
                    {
                        JsonElement jsonElement => ConvertJsonElement(jsonElement, targetType),
                        string stringValue => ParameterHelper.ConvertParameterValue(stringValue, targetType),
                        _ => Convert.ChangeType(value, targetType)
                    };

                    switch (member)
                    {
                        case PropertyInfo prop:
                            prop.SetValue(options, convertedValue);
                            Console.WriteLine($"Successfully set parameter '{cleanKey}' = {convertedValue}");
                            break;
                        case FieldInfo field:
                            field.SetValue(options, convertedValue);
                            Console.WriteLine($"Successfully set parameter '{cleanKey}' = {convertedValue}");
                            break;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Failed to set parameter '{cleanKey}'. Error: {ex.Message}, expected type: {member switch { PropertyInfo prop => prop.PropertyType.Name, FieldInfo field => field.FieldType.Name, _ => "Unknown" }}, received value: '{value}' (type: {value?.GetType().Name ?? "null"})");
                }
            }
        }


        private static object ConvertJsonElement(JsonElement element, Type targetType)
        {
            var nonNullableType = Nullable.GetUnderlyingType(targetType) ?? targetType;

            return nonNullableType switch
            {
                Type t when t == typeof(int) => element.ValueKind == JsonValueKind.String ? int.Parse(element.GetString()!) : element.GetInt32(),
                Type t when t == typeof(double) => element.ValueKind == JsonValueKind.String ? double.Parse(element.GetString()!) : element.GetDouble(),
                Type t when t == typeof(float) => element.ValueKind == JsonValueKind.String ? float.Parse(element.GetString()!) : element.GetSingle(),
                Type t when t == typeof(bool) => element.ValueKind == JsonValueKind.String ? bool.Parse(element.GetString()!) : element.GetBoolean(),
                Type t when t == typeof(decimal) => element.ValueKind == JsonValueKind.String ? decimal.Parse(element.GetString()!) : element.GetDecimal(),
                Type t when t == typeof(string) => element.GetString(),
                Type t when t.IsEnum => Enum.Parse(t, element.GetString()!, true),
                _ => Convert.ChangeType(element.GetRawText(), nonNullableType)
            };
        }
    }
}