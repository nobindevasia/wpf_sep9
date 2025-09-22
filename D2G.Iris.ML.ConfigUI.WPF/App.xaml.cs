using System;
using System.Windows;
using Microsoft.Extensions.DependencyInjection;
using D2G.Iris.ML.ConfigUI.WPF.Services;
using D2G.Iris.ML.ConfigUI.WPF.ViewModels;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Configuration;
using D2G.Iris.ML.Data;
using D2G.Iris.ML.Training;
using Microsoft.ML;

namespace D2G.Iris.ML.ConfigUI.WPF
{
    public partial class App : Application
    {
        private ServiceProvider? _serviceProvider;

        protected override void OnStartup(StartupEventArgs e)
        {
            var services = new ServiceCollection();

            services.AddSingleton<IConfigurationService, ConfigurationService>();
            services.AddSingleton<IDatabaseSchemaLoader, DatabaseSchemaLoader>();
            services.AddSingleton<IDialogService, DialogService>();
            services.AddSingleton<IConfigManager, ConfigManager>();
            services.AddSingleton<ISqlHandler>(provider => new SqlHandler("DefaultTable"));
            services.AddSingleton<IDataLoader, DatabaseDataLoader>();
            services.AddSingleton<IDataProcessor, DataProcessor>();
            services.AddSingleton<MLContext>();
            services.AddSingleton<IModelTrainerFactory, ModelTrainerFactory>();


            services.AddTransient<MainWindowViewModel>();
            services.AddTransient<GeneralSettingsViewModel>();
            services.AddTransient<DatabaseSettingsViewModel>();
            services.AddTransient<InputFieldsViewModel>();
            services.AddTransient<TrainingParametersViewModel>();
            services.AddTransient<DataBalancingViewModel>();
            services.AddTransient<FeatureEngineeringViewModel>();
            services.AddTransient<AutoMLSettingsViewModel>();
            services.AddTransient<TrainingLogsViewModel>();

            _serviceProvider = services.BuildServiceProvider();


            var mainWindowViewModel = _serviceProvider.GetRequiredService<MainWindowViewModel>();
            var mainWindow = new MainWindow(mainWindowViewModel);

            mainWindow.Show();

            base.OnStartup(e);
        }

        protected override void OnExit(ExitEventArgs e)
        {
            _serviceProvider?.Dispose();
            base.OnExit(e);
        }
    }
}