using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Enums;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace D2G.Iris.ML.ConfigUI.WPF.ViewModels
{
    public class AutoMLSettingsViewModel : BaseViewModel
    {
        private int _maxExperimentTimeInSeconds = 30;
        private int _maxModels = 10;
        private string _optimizingMetric = "Accuracy";
        private string _description = "AutoML will automatically try multiple algorithms and find the best performing model for your data.";

        public AutoMLSettingsViewModel()
        {
            UpdateAvailableMetrics(ModelType.BinaryClassification);
        }

        #region Properties

        public int MaxExperimentTimeInSeconds
        {
            get => _maxExperimentTimeInSeconds;
            set => SetProperty(ref _maxExperimentTimeInSeconds, System.Math.Max(1, System.Math.Min(3600, value)));
        }

        public int MaxModels
        {
            get => _maxModels;
            set => SetProperty(ref _maxModels, System.Math.Max(1, System.Math.Min(100, value)));
        }

        public string OptimizingMetric
        {
            get => _optimizingMetric;
            set => SetProperty(ref _optimizingMetric, value);
        }

        public string Description
        {
            get => _description;
            private set => SetProperty(ref _description, value);
        }

        public ObservableCollection<string> AvailableMetrics { get; } = new();
        #endregion


        public void SetConfiguration(AutoMLConfig? config)
        {
            if (config == null)
            {
                MaxExperimentTimeInSeconds = 30;
                MaxModels = 10;
                OptimizingMetric = "Accuracy";
                return;
            }
            MaxExperimentTimeInSeconds = config.MaxExperimentTimeInSeconds;
            MaxModels = config.MaxModels;
            OptimizingMetric = config.OptimizingMetric ?? "Accuracy";
        }

        public AutoMLConfig GetConfiguration()
        {
            return new AutoMLConfig
            {
                Enabled = true,
                MaxExperimentTimeInSeconds = MaxExperimentTimeInSeconds,
                MaxModels = MaxModels,
                OptimizingMetric = OptimizingMetric
            };
        }

        public void UpdateModelType(ModelType modelType)
        {
            UpdateAvailableMetrics(modelType);
        }

        private void UpdateAvailableMetrics(ModelType modelType)
        {
            AvailableMetrics.Clear();

            switch (modelType)
            {
                case ModelType.BinaryClassification:
                    foreach (var metric in new[] { "Accuracy", "AUC", "F1Score" })
                        AvailableMetrics.Add(metric);
                    OptimizingMetric = "Accuracy";
                    break;
                case ModelType.MultiClassClassification:
                    foreach (var metric in new[] { "MicroAccuracy", "MacroAccuracy" })
                        AvailableMetrics.Add(metric);
                    OptimizingMetric = "MicroAccuracy";
                    break;
                case ModelType.Regression:
                    foreach (var metric in new[] { "RSquared", "MeanAbsoluteError", "RootMeanSquaredError" })
                        AvailableMetrics.Add(metric);
                    OptimizingMetric = "RSquared";
                    break;
            }
        }
    }
}