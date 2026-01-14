"""
Data Preprocessing Script for California Housing Dataset

This script demonstrates comprehensive data cleaning and preprocessing techniques:
1. Loading raw data
2. Handling missing values
3. Feature engineering
4. Encoding categorical variables
5. Feature scaling
6. Train-test split
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def load_data(filepath):
    """Load the raw dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns\n")
    return df

def explore_data(df):
    """Explore the dataset and identify data quality issues"""
    print("=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    
    print("\n1. Dataset Info:")
    print(df.info())
    
    print("\n2. First few rows:")
    print(df.head())
    
    print("\n3. Statistical Summary:")
    print(df.describe())
    
    print("\n4. Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print("\n5. Data Types:")
    print(df.dtypes)
    
    print("\n6. Categorical Features:")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts())
    
    print("\n" + "=" * 60 + "\n")

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Strategy:
    - For total_bedrooms: Impute with median (numerical feature)
    - Document any other missing values found
    """
    print("=" * 60)
    print("HANDLING MISSING VALUES")
    print("=" * 60)
    
    df = df.copy()
    
    # Check for missing values
    missing_before = df.isnull().sum()
    print("\nMissing values before imputation:")
    print(missing_before[missing_before > 0])
    
    # Impute total_bedrooms with median
    if 'total_bedrooms' in df.columns and df['total_bedrooms'].isnull().any():
        median_bedrooms = df['total_bedrooms'].median()
        df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
        print(f"\nImputed {missing_before['total_bedrooms']} missing total_bedrooms values with median: {median_bedrooms}")
    
    # Verify no missing values remain
    missing_after = df.isnull().sum()
    print("\nMissing values after imputation:")
    print(missing_after[missing_after > 0] if missing_after.sum() > 0 else "No missing values!")
    
    print("\n" + "=" * 60 + "\n")
    return df

def feature_engineering(df):
    """
    Create new features from existing ones
    
    New features:
    -rooms_per_household: total_rooms / households
    - bedrooms_per_room: total_bedrooms / total_rooms
    - population_per_household: population / households
    """
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    df = df.copy()
    
    # Create new features
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    
    print("\nCreated 3 new features:")
    print("- rooms_per_household")
    print("- bedrooms_per_room") 
    print("- population_per_household")
    
    print("\nNew features statistics:")
    print(df[['rooms_per_household', 'bedrooms_per_room', 'population_per_household']].describe())
    
    print("\n" + "=" * 60 + "\n")
    return df

def handle_outliers(df):
    """
    Detect and optionally handle outliers
    
    For this dataset, we'll cap median_house_value at 500001 (known ceiling)
    """
    print("=" * 60)
    print("HANDLING OUTLIERS")
    print("=" * 60)
    
    df = df.copy()
    
    # The dataset has median_house_value capped at 500001
    # This is actually 500000+ but recorded as 500001
    outliers_count = (df['median_house_value'] >= 500001).sum()
    print(f"\nFound {outliers_count} properties with median_house_value >= 500001")
    print("These represent expensive properties (>$500K) and will be kept in the dataset")
    
    print("\n" + "=" * 60 + "\n")
    return df

def encode_categorical(df):
    """
    Encode categorical variables
    
    ocean_proximity will be one-hot encoded
    """
    print("=" * 60)
    print("ENCODING CATEGORICAL VARIABLES")
    print("=" * 60)
    
    df = df.copy()
    
    # One-hot encode ocean_proximity
    print("\nOne-hot encoding 'ocean_proximity'...")
    print(f"Categories: {df['ocean_proximity'].unique()}")
    
    df_encoded = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean')
    
    new_cols = [col for col in df_encoded.columns if col.startswith('ocean_')]
    print(f"\nCreated {len(new_cols)} new binary columns:")
    for col in new_cols:
        print(f"- {col}")
    
    print("\n" + "=" * 60 + "\n")
    return df_encoded

def split_features_target(df, target_column='median_house_value'):
    """
    Split dataset into features (X) and target (y)
    """
    print("=" * 60)
    print("SPLITTING FEATURES AND TARGET")
    print("=" * 60)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"\nTarget variable: {target_column}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {list(X.columns)}")
    
    print(f"\nTarget statistics:")
    print(y.describe())
    
    print("\n" + "=" * 60 + "\n")
    return X, y

def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler
    
    StandardScaler: (x - mean) / std
    """
    print("=" * 60)
    print("SCALING FEATURES")
    print("=" * 60)
    
    scaler = StandardScaler()
    
    # Fit on training data only to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nApplied StandardScaler to all features")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    print("\nScaler parameters (first 5 features):")
    for i, col in enumerate(X_train.columns[:5]):
        print(f"{col:30s} - mean: {scaler.mean_[i]:10.2f}, std: {scaler.scale_[i]:10.2f}")
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("\n" + "=" * 60 + "\n")
    return X_train_scaled, X_test_scaled, scaler

def save_processed_data(X_train, X_test, y_train, y_test, scaler, output_dir='data/cleaned'):
    """
    Save processed data and scaler
    """
    print("=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save data
    X_train.to_csv(output_path / 'X_train.csv', index=False)
    X_test.to_csv(output_path / 'X_test.csv', index=False)
    y_train.to_csv(output_path / 'y_train.csv', index=False, header=True)
    y_test.to_csv(output_path / 'y_test.csv', index=False, header=True)
    
    # Save scaler
    with open(output_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nSaved processed data to {output_dir}/")
    print(f"- X_train.csv: {X_train.shape}")
    print(f"- X_test.csv: {X_test.shape}")
    print(f"- y_train.csv: {y_train.shape}")
    print(f"- y_test.csv: {y_test.shape}")
    print(f"- scaler.pkl")
    
    print("\n" + "=" * 60 + "\n")

def main():
    """Main preprocessing pipeline"""
    print("\n" + "=" * 60)
    print("CALIFORNIA HOUSING DATA PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")
    
    # Step 1: Load data
    df = load_data('data/raw/housing.csv')
    
    # Step 2: Explore data
    explore_data(df)
    
    # Step 3: Handle missing values
    df_clean = handle_missing_values(df)
    
    # Step 4: Feature engineering
    df_engineered = feature_engineering(df_clean)
    
    # Step 5: Handle outliers
    df_outliers = handle_outliers(df_engineered)
    
    # Step 6: Encode categorical variables
    df_encoded = encode_categorical(df_outliers)
    
    # Step 7: Split features and target
    X, y = split_features_target(df_encoded)
    
    # Step 8: Train-test split
    print("=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Split ratio: 80/20")
    print("\n" + "=" * 60 + "\n")
    
    # Step 9: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 10: Save processed data
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    
    print("=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print("\nData is ready for model training.")
    print("Run 'python scripts/train.py' to train the model.\n")

if __name__ == "__main__":
    main()
