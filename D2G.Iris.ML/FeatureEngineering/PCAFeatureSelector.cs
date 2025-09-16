using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class PCAFeatureSelector : BaseFeatureSelector
    {
        public PCAFeatureSelector(MLContext mlContext)
            : base(mlContext)
        {
        }


        public override async Task<(IDataView transformedData, string[] selectedFeatures, string report)> SelectFeatures(
            MLContext mlContext,
            IDataView data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config)
        {
            InitializeReport("PCA");
            try
            {
                ValidatePcaConfiguration(config, candidateFeatures.Length);
                int k = config.NumberOfComponents;
                _report.AppendLine($"Applying PCA ({modelType}) with {k} components");

                if (modelType == ModelType.Regression)
                {
                    data = mlContext.Transforms
                            .Conversion.ConvertType(
                                outputColumnName: targetField,
                                inputColumnName: targetField,
                                outputKind: DataKind.Single)
                            .Fit(data)
                            .Transform(data);

                    var schemaFV = SchemaDefinition.Create(typeof(FeatureVectorFloat));
                    schemaFV[nameof(FeatureVectorFloat.Label)].ColumnName = targetField;

                    var rowsF = mlContext.Data.CreateEnumerable<FeatureVectorFloat>(
                        data,
                        reuseRowObject: false,
                        schemaDefinition: schemaFV)
                        .ToList();

                    if (rowsF.Count == 0 || rowsF[0].Features == null)
                        throw new InvalidOperationException("No valid feature data found");

                    int dim = rowsF[0].Features.Length;
                    var inputListF = rowsF
                        .Select(r => new PcaInputRowFloat { Label = r.Label, FeaturesArray = r.Features })
                        .ToList();

                    var schemaInF = SchemaDefinition.Create(typeof(PcaInputRowFloat));
                    schemaInF["FeaturesArray"].ColumnType =
                        new VectorDataViewType(NumberDataViewType.Single, dim);

                    var inputDataF = mlContext.Data.LoadFromEnumerable(inputListF, schemaInF);

                    var pipelineF = mlContext.Transforms
                        .NormalizeMinMax("NormalizedFeatures", "FeaturesArray")
                        .Append(mlContext.Transforms.ProjectToPrincipalComponents(
                            outputColumnName: "Features",
                            inputColumnName: "NormalizedFeatures",
                            rank: k));

                    var modelF = await Task.Run(() => pipelineF.Fit(inputDataF));
                    var transformedF = modelF.Transform(inputDataF);

                    var schemaOutF = SchemaDefinition.Create(typeof(PcaOutputRowFloat));
                    schemaOutF[nameof(PcaOutputRowFloat.Label)].ColumnName = "Label";

                    var resultF = mlContext.Data.CreateEnumerable<PcaOutputRowFloat>(
                        transformedF,
                        reuseRowObject: false,
                        schemaDefinition: schemaOutF)
                        .ToList();

                    var outputDataF = mlContext.Data.LoadFromEnumerable(resultF);

                    var pcaNamesF = Enumerable.Range(1, k)
                        .Select(i => $"PCA_Component_{i}")
                        .ToArray();

                    AddFeatureSelectionSummary(candidateFeatures.Length, k, pcaNamesF);
                    return (outputDataF, pcaNamesF, _report.ToString());
                }
                else
                {
                    
                    var labelColumnInfo = data.Schema.GetColumnOrNull(targetField);
                    if (labelColumnInfo != null && labelColumnInfo.Value.Type.RawType == typeof(bool))
                    {
                        
                        data = mlContext.Transforms
                            .Conversion.ConvertType(
                                outputColumnName: targetField,
                                inputColumnName: targetField,
                                outputKind: DataKind.Int64)
                            .Fit(data)
                            .Transform(data);
                    }

                    var schemaFV = SchemaDefinition.Create(typeof(FeatureVectorLong));
                    schemaFV[nameof(FeatureVectorLong.Label)].ColumnName = targetField;

                    var rowsL = mlContext.Data.CreateEnumerable<FeatureVectorLong>(
                        data,
                        reuseRowObject: false,
                        schemaDefinition: schemaFV)
                        .ToList();

                    if (rowsL.Count == 0 || rowsL[0].Features == null)
                        throw new InvalidOperationException("No valid feature data found");

                    int dim = rowsL[0].Features.Length;
                    var inputListL = rowsL
                        .Select(r => new PcaInputRowLong { Label = r.Label, FeaturesArray = r.Features })
                        .ToList();

                    var schemaInL = SchemaDefinition.Create(typeof(PcaInputRowLong));
                    schemaInL["FeaturesArray"].ColumnType =
                        new VectorDataViewType(NumberDataViewType.Single, dim);

                    var inputDataL = mlContext.Data.LoadFromEnumerable(inputListL, schemaInL);

                    var pipelineL = mlContext.Transforms
                        .NormalizeMinMax("NormalizedFeatures", "FeaturesArray")
                        .Append(mlContext.Transforms.ProjectToPrincipalComponents(
                            outputColumnName: "Features",
                            inputColumnName: "NormalizedFeatures",
                            rank: k));

                    var modelL = await Task.Run(() => pipelineL.Fit(inputDataL));
                    var transformedL = modelL.Transform(inputDataL);

                    var schemaOutL = SchemaDefinition.Create(typeof(PcaOutputRowLong));
                    schemaOutL[nameof(PcaOutputRowLong.Label)].ColumnName = "Label";

                    var resultL = mlContext.Data.CreateEnumerable<PcaOutputRowLong>(
                        transformedL,
                        reuseRowObject: false,
                        schemaDefinition: schemaOutL)
                        .ToList();

                    var outputDataL = mlContext.Data.LoadFromEnumerable(resultL);

                    var pcaNamesL = Enumerable.Range(1, k)
                        .Select(i => $"PCA_Component_{i}")
                        .ToArray();

                    AddFeatureSelectionSummary(candidateFeatures.Length, k, pcaNamesL);
                    return (outputDataL, pcaNamesL, _report.ToString());
                }
            }
            catch (Exception ex)
            {
                AddErrorToReport(ex);
                Console.WriteLine($"PCA Feature Selection Error: {ex.Message}");
                throw;
            }
        }
        private class FeatureVectorLong
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }
        private class PcaInputRowLong
        {
            public long Label { get; set; }
            public float[] FeaturesArray { get; set; }
        }
        private class PcaOutputRowLong
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }

        private class FeatureVectorFloat
        {
            [VectorType]
            public float[] Features { get; set; }
            public float Label { get; set; }
        }
        private class PcaInputRowFloat
        {
            public float Label { get; set; }
            public float[] FeaturesArray { get; set; }
        }
        private class PcaOutputRowFloat
        {
            [VectorType]
            public float[] Features { get; set; }
            public float Label { get; set; }
        }

        private void ValidatePcaConfiguration(FeatureEngineeringConfig config, int maxComponents)
        {
            base.ValidateConfiguration(config);

            if (config.NumberOfComponents <= 0 || config.NumberOfComponents > maxComponents)
            {
                _report.AppendLine($"Warning: Invalid number of components ({config.NumberOfComponents}). " +
                                  $"Using {Math.Min(maxComponents, 3)} instead.");
                config.NumberOfComponents = Math.Min(maxComponents, 3);
            }
        }
    }
}
