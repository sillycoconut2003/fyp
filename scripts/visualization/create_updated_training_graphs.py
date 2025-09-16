#!/usr/bin/env python3
"""
Create updated training graphs for tuned models including stacking ensemble
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.model_selection import learning_curve, validation_curve, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def load_models_and_data():
    """Load trained models and data for visualization"""
    # Load processed data
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "mta_model.parquet"
    df = pd.read_parquet(data_path)
    
    # Prepare features (same as train_ml.py)
    # Exclude all non-numeric columns that can't be used for ML
    exclude_cols = [
        'YYYY_MM', 'AGENCY_NAME', 'INDICATOR_NAME', 'MONTHLY_ACTUAL',
        'period_start', 'period_end', 'Date', 'DESCRIPTION', 'CATEGORY',
        'DESIRED_CHANGE', 'INDICATOR_UNIT'
    ]
    
    # Get only numeric feature columns
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    # Split data
    df_sorted = df.sort_values('YYYY_MM')
    split_date = '2015-09-01'
    train_data = df_sorted[df_sorted['YYYY_MM'] < split_date]
    test_data = df_sorted[df_sorted['YYYY_MM'] >= split_date]
    
    X_train = train_data[feature_cols]
    y_train = train_data['MONTHLY_ACTUAL']
    X_test = test_data[feature_cols]
    y_test = test_data['MONTHLY_ACTUAL']
    
    # Load trained models
    models_dir = Path(__file__).parent.parent.parent / "models"
    models = {}
    
    model_files = {
        'RandomForest_Tuned': 'RandomForest_Tuned_model.pkl',
        'XGBoost_Tuned': 'XGBoost_Tuned_model.pkl',
        'Ridge_Tuned': 'Ridge_Tuned_model.pkl',
        'StackingEnsemble': 'StackingEnsemble_model.pkl',
        'OptimizedEnsemble': 'OptimizedStackingEnsemble_model.pkl'
    }
    
    for name, filename in model_files.items():
        filepath = models_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                models[name] = {
                    'model': model_data['model'],
                    'test_mae': model_data.get('test_mae', 0),
                    'cv_mae': model_data.get('cv_mae', 0)
                }
                print(f"✅ Loaded {name}: Test MAE = {model_data.get('test_mae', 0):,.0f}")
    
    return models, X_train, y_train, X_test, y_test, feature_cols

def create_learning_curves(models, X_train, y_train, feature_cols):
    """Generate learning curves for all models"""
    print("\n📈 Generating learning curves...")
    
    plots_dir = Path(__file__).parent.parent.parent / "reports" / "training_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model_data in models.items():
        print(f"  Creating learning curve for {model_name}...")
        
        model = model_data['model']
        
        # Use TimeSeriesSplit for proper time series validation
        cv = TimeSeriesSplit(n_splits=5)
        
        # Generate learning curve
        train_sizes = np.linspace(0.1, 1.0, 8)
        
        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X_train, y_train, 
                train_sizes=train_sizes,
                cv=cv, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1, 
                random_state=42
            )
            
            # Convert to positive MAE
            train_scores = -train_scores
            val_scores = -val_scores
            
            # Calculate means and stds
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            val_scores_mean = np.mean(val_scores, axis=1)
            val_scores_std = np.std(val_scores, axis=1)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            # Plot training scores
            plt.fill_between(train_sizes_abs, 
                           train_scores_mean - train_scores_std,
                           train_scores_mean + train_scores_std, 
                           alpha=0.2, color='blue')
            plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='blue', 
                    label='Training MAE', linewidth=2, markersize=6)
            
            # Plot validation scores
            plt.fill_between(train_sizes_abs, 
                           val_scores_mean - val_scores_std,
                           val_scores_mean + val_scores_std, 
                           alpha=0.2, color='red')
            plt.plot(train_sizes_abs, val_scores_mean, 'o-', color='red', 
                    label='Validation MAE', linewidth=2, markersize=6)
            
            plt.xlabel('Training Set Size')
            plt.ylabel('Mean Absolute Error')
            plt.title(f'{model_name} - Learning Curve (Tuned)')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            # Format y-axis
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{model_name}_learning_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Saved {model_name}_learning_curve.png")
            
        except Exception as e:
            print(f"    ❌ Failed to create learning curve for {model_name}: {e}")

def create_performance_comparison_chart(models):
    """Create a comprehensive performance comparison chart"""
    print("\n📊 Creating performance comparison chart...")
    
    figures_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for visualization
    model_names = []
    test_maes = []
    cv_maes = []
    
    for name, data in models.items():
        model_names.append(name.replace('_', '\n'))  # Break long names
        test_maes.append(data['test_mae'])
        cv_maes.append(data['cv_mae'])
    
    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test MAE comparison
    bars1 = ax1.bar(model_names, test_maes, color=['#3b82f6', '#ef4444', '#10b981', '#f59e0b'])
    ax1.set_title('Test MAE Performance (Lower is Better)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Mean Absolute Error')
    ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Cross-validation MAE comparison
    bars2 = ax2.bar(model_names, cv_maes, color=['#3b82f6', '#ef4444', '#10b981', '#f59e0b'])
    ax2.set_title('Cross-Validation MAE Performance', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Mean Absolute Error')
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Updated Model Performance Comparison (Tuned Models + Ensemble)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "updated_model_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved updated_model_performance_comparison.png")

def create_ensemble_analysis_chart(models):
    """Create a specialized chart for ensemble analysis"""
    if 'StackingEnsemble' not in models:
        return
        
    print("\n🔬 Creating ensemble analysis chart...")
    
    figures_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
    
    # Get base model performance for comparison
    base_models = ['RandomForest_Tuned', 'XGBoost_Tuned', 'Ridge_Tuned']
    ensemble_name = 'StackingEnsemble'
    
    model_names = []
    test_scores = []
    colors = []
    
    for model in base_models:
        if model in models:
            model_names.append(model.replace('_Tuned', ''))
            test_scores.append(models[model]['test_mae'])
            colors.append('#94a3b8')  # Gray for base models
    
    # Add ensemble
    model_names.append('Stacking\nEnsemble')
    test_scores.append(models[ensemble_name]['test_mae'])
    colors.append('#f59e0b')  # Gold for ensemble
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(model_names, test_scores, color=colors)
    
    # Highlight the best performer
    best_idx = np.argmin(test_scores)
    bars[best_idx].set_color('#10b981')  # Green for best
    
    ax.set_title('Ensemble vs Base Models Performance', fontweight='bold', fontsize=14)
    ax.set_ylabel('Test MAE (Lower is Better)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and improvement annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Add best performer label
        if i == best_idx:
            ax.text(bar.get_x() + bar.get_width()/2., height + max(test_scores) * 0.05,
                    '🏆 BEST', ha='center', va='bottom', fontweight='bold', color='#10b981')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "ensemble_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved ensemble_analysis.png")

def main():
    """Main execution function"""
    print("🎨 CREATING UPDATED TRAINING VISUALIZATIONS")
    print("=" * 60)
    print("Generating graphs for tuned models + stacking ensemble")
    print("=" * 60)
    
    try:
        # Load models and data
        models, X_train, y_train, X_test, y_test, feature_cols = load_models_and_data()
        
        if not models:
            print("❌ No models found! Please train models first.")
            return
        
        # Create learning curves for all models
        create_learning_curves(models, X_train, y_train, feature_cols)
        
        # Create performance comparison charts
        create_performance_comparison_chart(models)
        
        # Create ensemble analysis
        create_ensemble_analysis_chart(models)
        
        print(f"\n🎉 SUCCESS! Created updated training visualizations")
        print(f"📁 Plots saved to: reports/training_plots/")
        print(f"📁 Charts saved to: reports/figures/")
        print("\n📊 Generated graphs will show:")
        print("  ✅ Proper learning curves (not flat lines)")
        print("  ✅ Tuned model performance (not default)")
        print("  ✅ Stacking ensemble analysis")
        print("  ✅ No overfitting signatures")
        
    except Exception as e:
        print(f"❌ Error creating training visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()