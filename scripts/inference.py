"""
Inference Script for California Housing Price Prediction

This script loads the trained model and runs inference on test examples,
visualizing predictions vs actual values with detailed feature analysis.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from model import HousingPriceModel

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_model(model_path='weights/best_model.pth', input_dim=16):
    """Load the trained model"""
    print("Loading trained model...")
    
    # Initialize model
    model = HousingPriceModel(input_dim=input_dim, dropout_rate=0.2)
    
    # Load weights
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"✓ Validation loss: {checkpoint['val_loss']:.2f}\n")
    
    return model

def load_test_data(data_dir='data/cleaned'):
    """Load test data and scalers"""
    print("Loading test data...")
    
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_test = pd.read_csv(f'{data_dir}/y_test.csv')
    
    # Load feature scaler
    with open(f'{data_dir}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load target scaler for inverse transformation
    with open(f'{data_dir}/target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    
    print(f"✓ Loaded {len(X_test)} test samples")
    print(f"✓ Features: {list(X_test.columns)}")
    print(f"✓ Target scaler loaded (mean=${target_scaler.mean_[0]:,.2f}, std=${target_scaler.scale_[0]:,.2f})\n")
    
    return X_test, y_test, scaler, target_scaler

def run_inference(model, X_test, target_scaler, num_samples=10):
    """Run inference on test samples"""
    print(f"Running inference on {num_samples} random samples...")
    
    # Randomly select samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_samples = X_test.iloc[indices]
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_samples.values)
    
    # Run inference
    with torch.no_grad():
        predictions_scaled = model(X_tensor).numpy()
    
    # Inverse transform to original scale (dollars)
    predictions = target_scaler.inverse_transform(predictions_scaled).flatten()
    
    print(f"✓ Inference complete\n")
    
    return indices, predictions

def create_prediction_comparison(indices, X_test, y_test, predictions, target_scaler, save_path='weights/inference_results.png'):
    """Create detailed visualization of predictions"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract actual values and inverse transform to original scale
    y_actual_scaled = y_test.values.flatten()[indices]
    y_actual = target_scaler.inverse_transform(y_actual_scaled.reshape(-1, 1)).flatten()
    
    # 1. Actual vs Predicted bar chart
    ax1 = fig.add_subplot(gs[0, :])
    x_pos = np.arange(len(indices))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, y_actual/1000, width, label='Actual Price', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, predictions/1000, width, label='Predicted Price', color='coral', alpha=0.8)
    
    ax1.set_xlabel('Sample Index', fontweight='bold', fontsize=12)
    ax1.set_ylabel('House Price ($1000s)', fontweight='bold', fontsize=12)
    ax1.set_title('Actual vs Predicted House Prices', fontweight='bold', fontsize=14, pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'#{i}' for i in range(len(indices))])
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.0f}K',
                    ha='center', va='bottom', fontsize=8)
    
    # 2. Prediction errors
    ax2 = fig.add_subplot(gs[1, 0])
    errors = predictions - y_actual
    colors = ['green' if e < 0 else 'red' for e in errors]
    bars = ax2.barh(range(len(errors)), errors/1000, color=colors, alpha=0.6)
    ax2.set_yticks(range(len(errors)))
    ax2.set_yticklabels([f'#{i}' for i in range(len(indices))])
    ax2.set_xlabel('Prediction Error ($1000s)', fontweight='bold')
    ax2.set_title('Prediction Errors\n(Red=Overestimated, Green=Underestimated)', fontweight='bold', fontsize=11)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Error distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(errors/1000, bins=15, color='purple', alpha=0.6, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Prediction Error ($1000s)', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Error Distribution', fontweight='bold', fontsize=11)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Percentage errors
    ax4 = fig.add_subplot(gs[1, 2])
    pct_errors = (errors / y_actual) * 100
    ax4.scatter(range(len(pct_errors)), pct_errors, s=100, c=colors, alpha=0.6, edgecolors='black')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Sample Index', fontweight='bold')
    ax4.set_ylabel('Percentage Error (%)', fontweight='bold')
    ax4.set_title('Percentage Prediction Errors', fontweight='bold', fontsize=11)
    ax4.set_xticks(range(len(pct_errors)))
    ax4.set_xticklabels([f'#{i}' for i in range(len(indices))])
    ax4.grid(alpha=0.3)
    
    # 5. Feature importance for first sample (show scaled values)
    ax5 = fig.add_subplot(gs[2, :])
    sample_idx = 0
    sample_features = X_test.iloc[indices[sample_idx]]
    
    # Get feature names and values
    feature_names = X_test.columns.tolist()
    feature_values = sample_features.values
    
    # Create horizontal bar chart
    colors_feat = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(feature_names)))
    bars = ax5.barh(feature_names, feature_values, color=colors_feat, alpha=0.7, edgecolor='black')
    
    ax5.set_xlabel('Scaled Feature Value (z-score)', fontweight='bold', fontsize=12)
    ax5.set_title(f'Feature Values for Sample #{sample_idx}\nActual: ${y_actual[sample_idx]:,.0f} | Predicted: ${predictions[sample_idx]:,.0f} | Error: ${errors[sample_idx]:,.0f}', 
                  fontweight='bold', fontsize=12, pad=15)
    ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax5.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, feature_values)):
        ax5.text(val, i, f' {val:.2f}', va='center', fontsize=8)
    
    plt.suptitle('🏠 California Housing Price Prediction - Inference Results 🏠', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {save_path}")
    plt.close()

def create_detailed_comparison_table(indices, X_test, y_test, predictions, target_scaler, save_path='weights/inference_table.png'):
    """Create a detailed table comparing predictions"""
    
    # Get original (unscaled) feature names for better readability
    feature_names = ['Longitude', 'Latitude', 'House Age', 'Total Rooms', 
                    'Total Bedrooms', 'Population', 'Households', 'Median Income',
                    'Rooms/HH', 'Bed/Room', 'Pop/HH', 
                    'Ocean <1H', 'Ocean Inland', 'Ocean Island', 'Ocean Near Bay', 'Ocean Near Ocean']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table - inverse transform actual values
    y_actual_scaled = y_test.values.flatten()[indices]
    y_actual = target_scaler.inverse_transform(y_actual_scaled.reshape(-1, 1)).flatten()
    errors = predictions - y_actual
    pct_errors = (errors / y_actual) * 100
    
    # Create table data
    table_data = []
    for i, idx in enumerate(indices):
        row = [
            f'#{i}',
            f'${y_actual[i]:,.0f}',
            f'${predictions[i]:,.0f}',
            f'${errors[i]:,.0f}',
            f'{pct_errors[i]:.1f}%'
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Sample', 'Actual Price', 'Predicted Price', 'Error ($)', 'Error (%)'],
                    cellLoc='center',
                    loc='center',
                    colColours=['lightblue']*5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color code error cells
    for i in range(len(table_data)):
        # Color error cells based on magnitude
        error_val = errors[i]
        if abs(error_val) < 50000:
            color = 'lightgreen'
        elif abs(error_val) < 100000:
            color = 'lightyellow'
        else:
            color = 'lightcoral'
        table[(i+1, 3)].set_facecolor(color)
        table[(i+1, 4)].set_facecolor(color)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('steelblue')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Detailed Prediction Comparison Table', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison table to {save_path}")
    plt.close()

def print_summary_statistics(indices, y_test, predictions, target_scaler):
    """Print summary statistics of predictions"""
    
    # Inverse transform actual values to original scale
    y_actual_scaled = y_test.values.flatten()[indices]
    y_actual = target_scaler.inverse_transform(y_actual_scaled.reshape(-1, 1)).flatten()
    errors = predictions - y_actual
    pct_errors = (errors / y_actual) * 100
    
    print("="*60)
    print("INFERENCE SUMMARY STATISTICS")
    print("="*60)
    print(f"\nNumber of samples: {len(indices)}")
    print(f"\nPrice Statistics:")
    print(f"  Actual prices:    ${y_actual.min():,.0f} - ${y_actual.max():,.0f}")
    print(f"  Predicted prices: ${predictions.min():,.0f} - ${predictions.max():,.0f}")
    print(f"\nError Statistics:")
    print(f"  Mean Error:       ${errors.mean():,.0f}")
    print(f"  Median Error:     ${np.median(errors):,.0f}")
    print(f"  Std Dev:          ${errors.std():,.0f}")
    print(f"  MAE:              ${np.abs(errors).mean():,.0f}")
    print(f"  RMSE:             ${np.sqrt((errors**2).mean()):,.0f}")
    print(f"\nPercentage Errors:")
    print(f"  Mean:             {pct_errors.mean():.1f}%")
    print(f"  Median:           {np.median(pct_errors):.1f}%")
    print(f"  Range:            {pct_errors.min():.1f}% to {pct_errors.max():.1f}%")
    print("\n" + "="*60 + "\n")

def print_individual_predictions(indices, X_test, y_test, predictions, target_scaler, num_to_show=5):
    """Print detailed information for individual predictions"""
    
    # Inverse transform actual values to original scale
    y_actual_scaled = y_test.values.flatten()[indices]
    y_actual = target_scaler.inverse_transform(y_actual_scaled.reshape(-1, 1)).flatten()
    errors = predictions - y_actual
    
    print("="*60)
    print(f"DETAILED PREDICTIONS (First {num_to_show} samples)")
    print("="*60 + "\n")
    
    for i in range(min(num_to_show, len(indices))):
        print(f"Sample #{i}:")
        print(f"  Actual Price:     ${y_actual[i]:>12,.2f}")
        print(f"  Predicted Price:  ${predictions[i]:>12,.2f}")
        print(f"  Error:            ${errors[i]:>12,.2f} ({(errors[i]/y_actual[i]*100):>6.1f}%)")
        
        # Show a few key features
        sample = X_test.iloc[indices[i]]
        print(f"  Key Features (scaled):")
        print(f"    Median Income:   {sample['median_income']:>7.2f}")
        print(f"    Rooms per HH:    {sample['rooms_per_household']:>7.2f}")
        print(f"    House Age:       {sample['housing_median_age']:>7.2f}")
        print()
    
    print("="*60 + "\n")

def main():
    """Main inference pipeline"""
    print("\n" + "="*60)
    print("CALIFORNIA HOUSING PRICE PREDICTION - INFERENCE")
    print("="*60 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load model and data
    model = load_model()
    X_test, y_test, scaler, target_scaler = load_test_data()
    
    # Run inference on random samples
    num_samples = 10
    indices, predictions = run_inference(model, X_test, target_scaler, num_samples=num_samples)
    
    # Print detailed predictions
    print_individual_predictions(indices, X_test, y_test, predictions, target_scaler, num_to_show=5)
    
    # Print summary statistics
    print_summary_statistics(indices, y_test, predictions, target_scaler)
    
    # Create visualizations
    print("Creating visualizations...")
    create_prediction_comparison(indices, X_test, y_test, predictions, target_scaler)
    create_detailed_comparison_table(indices, X_test, y_test, predictions, target_scaler)
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - weights/inference_results.png")
    print("  - weights/inference_table.png")
    print("\nCheck these files for detailed visualizations!\n")

if __name__ == "__main__":
    main()
