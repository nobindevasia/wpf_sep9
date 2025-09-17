using System;
using System.ComponentModel;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.ConfigUI.WPF.ViewModels
{
    public class DataProcessingPipelineViewModel : BaseViewModel
    {
        private readonly DataBalancingViewModel _dataBalancingViewModel;
        private readonly FeatureEngineeringViewModel _featureEngineeringViewModel;
        private string _intermediateResultsTableName = "";

        public DataProcessingPipelineViewModel()
        {
            _dataBalancingViewModel = new DataBalancingViewModel();
            _featureEngineeringViewModel = new FeatureEngineeringViewModel();

            _dataBalancingViewModel.ExecutionOrder = 1;
            _featureEngineeringViewModel.ExecutionOrder = 2;

            _dataBalancingViewModel.PropertyChanged += OnChildViewModelPropertyChanged;
            _featureEngineeringViewModel.PropertyChanged += OnChildViewModelPropertyChanged;
        }

        #region Properties

        public DataBalancingViewModel DataBalancing => _dataBalancingViewModel;
        public FeatureEngineeringViewModel FeatureEngineering => _featureEngineeringViewModel;

        public bool IsDataBalancingEnabled
        {
            get => _dataBalancingViewModel.IsEnabled;
            set
            {
                if (value != _dataBalancingViewModel.IsEnabled)
                {
                    _dataBalancingViewModel.SelectedMethod = value
                        ? DataBalanceMethod.SMOTE
                        : DataBalanceMethod.None;
                }
            }
        }

        public bool IsFeatureEngineeringEnabled
        {
            get => _featureEngineeringViewModel.SelectedMethod != FeatureSelectionMethod.None;
            set
            {
                var currentlyEnabled = _featureEngineeringViewModel.SelectedMethod != FeatureSelectionMethod.None;
                if (value != currentlyEnabled)
                {
                    _featureEngineeringViewModel.SelectedMethod = value
                        ? FeatureSelectionMethod.Correlation
                        : FeatureSelectionMethod.None;
                }
            }
        }

        public int DataBalancingExecutionOrder
        {
            get => _dataBalancingViewModel.ExecutionOrder;
            set => _dataBalancingViewModel.ExecutionOrder = value;
        }

        public int FeatureEngineeringExecutionOrder
        {
            get => _featureEngineeringViewModel.ExecutionOrder;
            set => _featureEngineeringViewModel.ExecutionOrder = value;
        }

        public string DataBalancingExecutionOrderString
        {
            get => _dataBalancingViewModel.ExecutionOrder.ToString();
            set
            {
                if (int.TryParse(value, out int order))
                {
                    _dataBalancingViewModel.ExecutionOrder = order;
                }
            }
        }

        public string FeatureEngineeringExecutionOrderString
        {
            get => _featureEngineeringViewModel.ExecutionOrder.ToString();
            set
            {
                if (int.TryParse(value, out int order))
                {
                    _featureEngineeringViewModel.ExecutionOrder = order;
                }
            }
        }

        public string IntermediateResultsTableName
        {
            get => _intermediateResultsTableName;
            set => SetProperty(ref _intermediateResultsTableName, value);
        }

        #endregion

        #region Private Methods

        private void OnChildViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
        {

            if (e.PropertyName == nameof(DataBalancingViewModel.IsEnabled))
            {
                OnPropertyChanged(nameof(IsDataBalancingEnabled));
            }
            else if (e.PropertyName == nameof(FeatureEngineeringViewModel.SelectedMethod))
            {
                OnPropertyChanged(nameof(IsFeatureEngineeringEnabled));
            }
            else if (e.PropertyName == nameof(DataBalancingViewModel.ExecutionOrder))
            {
                OnPropertyChanged(nameof(DataBalancingExecutionOrder));
                OnPropertyChanged(nameof(DataBalancingExecutionOrderString));
            }
            else if (e.PropertyName == nameof(FeatureEngineeringViewModel.ExecutionOrder))
            {
                OnPropertyChanged(nameof(FeatureEngineeringExecutionOrder));
                OnPropertyChanged(nameof(FeatureEngineeringExecutionOrderString));
            }
        }


        #endregion

        #region Public Methods

        public void LoadFromConfig(ModelConfig config)
        {
            _dataBalancingViewModel.SetConfiguration(config.DataBalancing);
            _featureEngineeringViewModel.SetConfiguration(config.FeatureEngineering);
            IntermediateResultsTableName = config.Database?.OutputTableName ?? "";
        }

        public void SaveToConfig(ModelConfig config)
        {
            config.DataBalancing = _dataBalancingViewModel.GetConfiguration();
            config.FeatureEngineering = _featureEngineeringViewModel.GetConfiguration();
            if (config.Database != null)
            {
                config.Database.OutputTableName = IntermediateResultsTableName;
            }
        }
        
        public void ResetToDefaults()
        {
            _dataBalancingViewModel.SetConfiguration(null);
            _featureEngineeringViewModel.SetConfiguration(null);
            IntermediateResultsTableName = "";
        }

        #endregion

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _dataBalancingViewModel.PropertyChanged -= OnChildViewModelPropertyChanged;
                _featureEngineeringViewModel.PropertyChanged -= OnChildViewModelPropertyChanged;

                _dataBalancingViewModel.Dispose();
                _featureEngineeringViewModel.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}