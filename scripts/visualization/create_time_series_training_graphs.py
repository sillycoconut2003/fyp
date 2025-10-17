#!/usr/bin/env python3
"""
Create comprehensive training visualizations for time series models
Generates forecast plots, residuals analysis, and diagnostics for Prophet and SARIMA models
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import pickle
import warnings
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import PROCESSED, COLS
from eval import rmse, mae, mape

warnings.filterwarnings('ignore')

def load_time_series_models():
    """Load trained time series models"""
    models_dir = Path(__file__).parent.parent.parent / "models" / "time_series"
    
    prophet_models = {}
    sarima_models = {}
    
    # Load Prophet models
    prophet_path = models_dir / "prophet_models.pkl"
    if prophet_path.exists():
        with open(prophet_path, 'rb') as f:
            prophet_models = pickle.load(f)
        print(f"✅ Loaded {len(prophet_models)} Prophet models")
    
    # Load SARIMA models
    sarima_path = models_dir / "sarima_models.pkl"
    if sarima_path.exists():
        with open(sarima_path, 'rb') as f:
            sarima_models = pickle.load(f)
        print(f"✅ Loaded {len(sarima_models)} SARIMA models")
    
    return prophet_models, sarima_models

def load_processed_data():
    """Load processed data for time series visualization"""
    print(f"📊 Loading processed data from: {PROCESSED}")
    df = pd.read_parquet(PROCESSED)
    print(f"✓ Loaded dataset: {df.shape[0]} records, {df.shape[1]} columns")
    return df

def prepare_series_data(df, agency, indicator):
    """Prepare time series data for specific agency-indicator combination"""
    series_df = df[
        (df[COLS['agency']] == agency) & 
        (df[COLS['indicator']] == indicator)
    ].sort_values('YYYY_MM').copy()
    
    # Remove missing target values
    series_df = series_df.dropna(subset=['MONTHLY_ACTUAL'])
    return series_df

def split_series_temporal(series_df, test_months=12):
    """Split time series maintaining temporal order"""
    min_training_months = 12
    min_total_months = test_months + min_training_months
    
    if len(series_df) < min_total_months:
        return None, None
    
    split_idx = len(series_df) - test_months
    train_df = series_df.iloc[:split_idx].copy()
    test_df = series_df.iloc[split_idx:].copy()
    
    return train_df, test_df

def create_prophet_forecast_plot(model_data, train_df, test_df, series_key):
    """Create Prophet forecast visualization with actual vs predicted"""
    try:
        model = model_data['model']
        agency = model_data['agency']
        indicator = model_data['indicator']
        
        # Create comprehensive forecast plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Prepare Prophet data format
        prophet_train = train_df[['YYYY_MM', 'MONTHLY_ACTUAL']].copy()
        prophet_train.columns = ['ds', 'y']
        
        # Generate forecast for full period (train + test)
        future = model.make_future_dataframe(periods=len(test_df), freq='MS')
        forecast = model.predict(future)
        
        # Plot 1: Full forecast with components
        ax1.plot(prophet_train['ds'], prophet_train['y'], 'ko-', 
                label='Training Data', markersize=4, alpha=0.7)
        ax1.plot(test_df['YYYY_MM'], test_df['MONTHLY_ACTUAL'], 'ro-',
                label='Actual Test Data', markersize=4, alpha=0.7)
        ax1.plot(forecast['ds'], forecast['yhat'], 'b-',
                label='Prophet Forecast', linewidth=2)
        ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                        alpha=0.3, color='blue', label='Prediction Interval')
        
        # Add vertical line at train/test split
        split_date = test_df['YYYY_MM'].iloc[0]
        ax1.axvline(split_date, color='gray', linestyle='--', alpha=0.7, label='Train/Test Split')
        
        ax1.set_title(f'Prophet Forecast: {agency} - {indicator}', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Monthly Actual Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Residuals analysis (test period only)
        test_predictions = forecast.tail(len(test_df))['yhat'].values
        test_actual = test_df['MONTHLY_ACTUAL'].values
        residuals = test_actual - test_predictions
        
        ax2.scatter(test_predictions, residuals, alpha=0.6, s=40)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals (Actual - Predicted)')
        ax2.set_title('Residuals Analysis (Test Period)', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add performance metrics as text
        test_mae = mae(test_actual, test_predictions)
        test_rmse = rmse(test_actual, test_predictions)
        test_mape = mape(test_actual, test_predictions)
        
        metrics_text = f'Test Metrics:\nMAE: {test_mae:.2f}\nRMSE: {test_rmse:.2f}\nMAPE: {test_mape:.2f}%'
        ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
                verticalalignment='top')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"    ❌ Failed to create Prophet plot for {series_key}: {e}")
        return None

def create_sarima_forecast_plot(model_data, train_df, test_df, series_key):
    """Create SARIMA forecast visualization with diagnostics"""
    try:
        model = model_data['model']
        agency = model_data['agency']
        indicator = model_data['indicator']
        best_params = model_data.get('best_params', (1, 1, 1))
        
        # Create comprehensive SARIMA plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare time series data
        ts_data = train_df.set_index('YYYY_MM')['MONTHLY_ACTUAL']
        
        # Generate forecast
        forecast_steps = len(test_df)
        forecast = model.forecast(steps=forecast_steps)
        forecast_dates = test_df['YYYY_MM'].values
        
        # Plot 1: Forecast vs Actual
        ax1.plot(train_df['YYYY_MM'], train_df['MONTHLY_ACTUAL'], 'ko-',
                label='Training Data', markersize=3, alpha=0.7)
        ax1.plot(test_df['YYYY_MM'], test_df['MONTHLY_ACTUAL'], 'ro-',
                label='Actual Test Data', markersize=4, alpha=0.7)
        ax1.plot(forecast_dates, forecast.values, 'b-',
                label=f'SARIMA{best_params} Forecast', linewidth=2)
        
        # Add confidence intervals if available
        try:
            forecast_ci = model.get_forecast(steps=forecast_steps).conf_int()
            ax1.fill_between(forecast_dates, 
                           forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                           alpha=0.3, color='blue', label='Confidence Interval')
        except:
            pass
        
        # Add vertical line at train/test split
        split_date = test_df['YYYY_MM'].iloc[0]
        ax1.axvline(split_date, color='gray', linestyle='--', alpha=0.7, label='Train/Test Split')
        
        ax1.set_title(f'SARIMA Forecast: {agency} - {indicator}', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Monthly Actual Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Residuals analysis
        test_predictions = forecast.values
        test_actual = test_df['MONTHLY_ACTUAL'].values
        residuals = test_actual - test_predictions
        
        ax2.scatter(test_predictions, residuals, alpha=0.6, s=40, color='orange')
        ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Analysis (Test Period)', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model diagnostics - Standardized residuals
        try:
            standardized_residuals = model.resid / np.std(model.resid)
            ax3.plot(standardized_residuals, 'o-', markersize=3, alpha=0.6)
            ax3.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(2, color='orange', linestyle=':', alpha=0.5, label='±2σ')
            ax3.axhline(-2, color='orange', linestyle=':', alpha=0.5)
            ax3.set_title('Standardized Residuals (Training)', fontweight='bold', fontsize=12)
            ax3.set_ylabel('Standardized Residuals')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        except:
            ax3.text(0.5, 0.5, 'Diagnostics\nNot Available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Model Diagnostics', fontweight='bold', fontsize=12)
        
        # Plot 4: QQ plot for residuals normality
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.set_title('Q-Q Plot: Residuals Normality Check', fontweight='bold', fontsize=12)
            ax4.grid(True, alpha=0.3)
        except:
            ax4.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            ax4.set_title('Residuals Distribution', fontweight='bold', fontsize=12)
            ax4.set_xlabel('Residuals')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        # Add performance metrics
        test_mae = mae(test_actual, test_predictions)
        test_rmse = rmse(test_actual, test_predictions)
        test_mape = mape(test_actual, test_predictions)
        
        metrics_text = f'Model: SARIMA{best_params}\nTest Metrics:\nMAE: {test_mae:.2f}\nRMSE: {test_rmse:.2f}\nMAPE: {test_mape:.2f}%'
        fig.text(0.02, 0.98, metrics_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"    ❌ Failed to create SARIMA plot for {series_key}: {e}")
        return None

def create_time_series_comparison_plot(prophet_models, sarima_models, df):
    """Create aggregate performance comparison for time series models"""
    print("\n📊 Creating time series models comparison plot...")
    
    # Calculate aggregate metrics
    prophet_metrics = []
    sarima_metrics = []
    
    for series_key in prophet_models:
        if series_key in sarima_models:
            prophet_data = prophet_models[series_key]
            sarima_data = sarima_models[series_key]
            
            prophet_metrics.append({
                'mae': prophet_data['test_mae'],
                'rmse': prophet_data['test_rmse'], 
                'mape': prophet_data['test_mape']
            })
            
            sarima_metrics.append({
                'mae': sarima_data['test_mae'],
                'rmse': sarima_data['test_rmse'],
                'mape': sarima_data['test_mape']
            })
    
    if not prophet_metrics or not sarima_metrics:
        print("    ❌ No matching models found for comparison")
        return None
    
    # Calculate averages
    prophet_avg_mae = np.mean([m['mae'] for m in prophet_metrics])
    prophet_avg_rmse = np.mean([m['rmse'] for m in prophet_metrics])
    prophet_avg_mape = np.mean([m['mape'] for m in prophet_metrics])
    
    sarima_avg_mae = np.mean([m['mae'] for m in sarima_metrics])
    sarima_avg_rmse = np.mean([m['rmse'] for m in sarima_metrics])
    sarima_avg_mape = np.mean([m['mape'] for m in sarima_metrics])
    
    # Create comparison chart
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    models = ['Prophet\nTuned', 'SARIMA\nTuned']
    colors = ['#3b82f6', '#ef4444']
    
    # MAE comparison
    mae_values = [prophet_avg_mae, sarima_avg_mae]
    bars1 = ax1.bar(models, mae_values, color=colors, alpha=0.7)
    ax1.set_title('Average MAE Comparison', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Mean Absolute Error')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    for bar, value in zip(bars1, mae_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values) * 0.01,
                f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE comparison
    rmse_values = [prophet_avg_rmse, sarima_avg_rmse]
    bars2 = ax2.bar(models, rmse_values, color=colors, alpha=0.7)
    ax2.set_title('Average RMSE Comparison', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Root Mean Squared Error')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    for bar, value in zip(bars2, rmse_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values) * 0.01,
                f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # MAPE comparison
    mape_values = [prophet_avg_mape, sarima_avg_mape]
    bars3 = ax3.bar(models, mape_values, color=colors, alpha=0.7)
    ax3.set_title('Average MAPE Comparison', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Mean Absolute Percentage Error (%)')
    
    for bar, value in zip(bars3, mape_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mape_values) * 0.01,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add overall statistics
    n_series = len(prophet_metrics)
    fig.suptitle(f'Time Series Models Performance Comparison\n({n_series} Series Evaluated)', 
                fontsize=16, fontweight='bold')
    
    # Highlight better performer
    if prophet_avg_mae < sarima_avg_mae:
        bars1[0].set_color('#10b981')  # Green for better
        bars2[0].set_color('#10b981')
        bars3[0].set_color('#10b981')
    else:
        bars1[1].set_color('#10b981')
        bars2[1].set_color('#10b981') 
        bars3[1].set_color('#10b981')
    
    plt.tight_layout()
    return fig

def generate_sample_time_series_plots(prophet_models, sarima_models, df, max_plots=5):
    """Generate sample time series training plots for visualization"""
    print(f"\n📈 Generating sample time series training plots (max {max_plots})...")
    
    plots_dir = Path(__file__).parent.parent.parent / "reports" / "training_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    created_plots = []
    plot_count = 0
    
    # Get series keys that have both Prophet and SARIMA models
    common_series = set(prophet_models.keys()) & set(sarima_models.keys())
    
    for series_key in list(common_series)[:max_plots]:
        agency, indicator = series_key.split('|')
        print(f"  Creating plots for: {agency} - {indicator}")
        
        # Prepare series data
        series_df = prepare_series_data(df, agency, indicator)
        train_df, test_df = split_series_temporal(series_df)
        
        if train_df is None or test_df is None:
            continue
        
        # Create Prophet plot
        prophet_fig = create_prophet_forecast_plot(
            prophet_models[series_key], train_df, test_df, series_key
        )
        if prophet_fig:
            safe_filename = series_key.replace('|', '_').replace(' ', '_').replace('/', '_')
            prophet_path = plots_dir / f"Prophet_{safe_filename}_forecast.png"
            prophet_fig.savefig(prophet_path, dpi=300, bbox_inches='tight')
            plt.close(prophet_fig)
            created_plots.append(f"Prophet_{safe_filename}_forecast.png")
            print(f"    ✅ Created Prophet plot: {prophet_path.name}")
        
        # Create SARIMA plot
        sarima_fig = create_sarima_forecast_plot(
            sarima_models[series_key], train_df, test_df, series_key
        )
        if sarima_fig:
            safe_filename = series_key.replace('|', '_').replace(' ', '_').replace('/', '_')
            sarima_path = plots_dir / f"SARIMA_{safe_filename}_diagnostics.png"
            sarima_fig.savefig(sarima_path, dpi=300, bbox_inches='tight')
            plt.close(sarima_fig)
            created_plots.append(f"SARIMA_{safe_filename}_diagnostics.png")
            print(f"    ✅ Created SARIMA plot: {sarima_path.name}")
        
        plot_count += 1
        if plot_count >= max_plots:
            break
    
    return created_plots

def create_time_series_aggregate_performance_table():
    """Create a summary table of time series performance metrics"""
    print("\n📊 Creating time series aggregate performance summary...")
    
    # Define the aggregate metrics (from your conversation context)
    performance_data = {
        'Model': ['Prophet Baseline', 'Prophet Tuned', 'SARIMA Baseline', 'SARIMA Tuned'],
        'MAE': [78344.85, 75922.55, 93580.65, 94511.65],
        'RMSE': [91059.32, 88396.45, 108944.22, 109944.33],
        'MAPE (%)': [9.37, 8.94, 11.18, 11.30],
        'Series Count': [132, 132, 132, 132],
        'Improvement': ['Baseline', '+3.1%', 'Baseline', '-1.0%']
    }
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Performance table
    df_performance = pd.DataFrame(performance_data)
    
    # Create table visualization
    table_data = []
    for _, row in df_performance.iterrows():
        table_data.append([
            row['Model'],
            f"{row['MAE']:,.0f}",
            f"{row['RMSE']:,.0f}",
            f"{row['MAPE (%)']:.2f}%",
            row['Improvement']
        ])
    
    table = ax1.table(cellText=table_data,
                     colLabels=['Model', 'MAE', 'RMSE', 'MAPE', 'Improvement'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(df_performance) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#3b82f6')
                cell.set_text_props(weight='bold', color='white')
            elif 'Tuned' in table_data[i-1][0] and table_data[i-1][4] != 'Baseline':
                if '+' in table_data[i-1][4]:
                    cell.set_facecolor('#dcfce7')  # Light green for improvements
                else:
                    cell.set_facecolor('#fee2e2')  # Light red for degradations
    
    ax1.axis('off')
    ax1.set_title('Time Series Models: Aggregate Performance Summary\n(132 Series Each)', 
                 fontweight='bold', fontsize=14, pad=20)
    
    # Performance comparison chart
    prophet_mae = [78344.85, 75922.55]
    sarima_mae = [93580.65, 94511.65]
    x_pos = np.arange(2)
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, prophet_mae, width, label='Prophet', color='#3b82f6', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, sarima_mae, width, label='SARIMA', color='#ef4444', alpha=0.7)
    
    ax2.set_xlabel('Model Configuration')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Prophet vs SARIMA: Before/After Tuning', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Baseline', 'Tuned'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height/1000:.0f}K', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """Main execution function"""
    print("🎨 CREATING TIME SERIES TRAINING VISUALIZATIONS")
    print("=" * 70)
    print("Generating comprehensive plots for Prophet and SARIMA models")
    print("=" * 70)
    
    try:
        # Load time series models
        prophet_models, sarima_models = load_time_series_models()
        
        if not prophet_models and not sarima_models:
            print("❌ No time series models found! Please train time series models first.")
            print("   Run: python src/train_ts.py")
            return
        
        # Load processed data
        df = load_processed_data()
        
        # Create directories
        plots_dir = Path(__file__).parent.parent.parent / "reports" / "training_plots"
        figures_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
        plots_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        # Generate sample individual time series plots
        sample_plots = generate_sample_time_series_plots(prophet_models, sarima_models, df, max_plots=5)
        created_files.extend(sample_plots)
        
        # Create aggregate comparison plot
        if prophet_models and sarima_models:
            comparison_fig = create_time_series_comparison_plot(prophet_models, sarima_models, df)
            if comparison_fig:
                comparison_path = figures_dir / "time_series_models_comparison.png"
                comparison_fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
                plt.close(comparison_fig)
                created_files.append("time_series_models_comparison.png")
                print(f"✅ Created aggregate comparison: {comparison_path.name}")
        
        # Create performance summary table
        summary_fig = create_time_series_aggregate_performance_table()
        summary_path = figures_dir / "time_series_performance_summary.png"
        summary_fig.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close(summary_fig)
        created_files.append("time_series_performance_summary.png")
        print(f"✅ Created performance summary: {summary_path.name}")
        
        # Final summary
        print(f"\n🎉 SUCCESS! Created {len(created_files)} time series visualizations")
        print("=" * 50)
        print("📁 Training plots saved to: reports/training_plots/")
        print("📁 Summary charts saved to: reports/figures/")
        print("\n📊 Generated visualizations include:")
        print("  ✅ Individual Prophet forecast plots with residuals")
        print("  ✅ SARIMA diagnostic plots with model validation")
        print("  ✅ Aggregate Prophet vs SARIMA comparison")
        print("  ✅ Performance summary table with improvements")
        print("  ✅ Time series specific training analysis")
        
        print(f"\n📋 Files created:")
        for file in created_files:
            print(f"  • {file}")
        
        return created_files
        
    except Exception as e:
        print(f"❌ Error creating time series training visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()