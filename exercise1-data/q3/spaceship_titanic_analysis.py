"""
Spaceship Titanic Dataset Analysis and Preprocessing
Exercise 3: Neural Network Data Preparation

This script performs comprehensive data analysis and preprocessing
for the Spaceship Titanic dataset, preparing it for neural network training
with tanh activation functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load the dataset and perform initial exploration."""
    print("="*60)
    print("SPACESHIP TITANIC DATASET ANALYSIS")
    print("="*60)
    
    # Load the data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    return train_df, test_df

def describe_dataset(df):
    """Describe the dataset's objective and features."""
    print(f"\n🎯 DATASET OBJECTIVE:")
    print("The 'Transported' column represents whether a passenger was transported")
    print("to an alternate dimension during the Spaceship Titanic's collision with")
    print("a spacetime anomaly. This is a binary classification problem.")
    
    print(f"\n📋 DATASET FEATURES:")
    print(f"Total columns: {len(df.columns)}")
    
    # Identify feature types
    numerical_features = []
    categorical_features = []
    
    for col in df.columns:
        if col in ['PassengerId', 'Name', 'Transported']:
            continue
        elif df[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
        else:
            categorical_features.append(col)
    
    print(f"\n🔢 NUMERICAL FEATURES ({len(numerical_features)}):")
    for feat in numerical_features:
        print(f"  • {feat}")
    
    print(f"\n🏷️  CATEGORICAL FEATURES ({len(categorical_features)}):")
    for feat in categorical_features:
        print(f"  • {feat}")
    
    return numerical_features, categorical_features

def investigate_missing_values(df):
    """Investigate missing values in the dataset."""
    print(f"\n🔍 MISSING VALUES ANALYSIS:")
    
    missing_info = []
    total_rows = len(df)
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_percentage = (missing_count / total_rows) * 100
        if missing_count > 0:
            missing_info.append({
                'Column': col,
                'Missing Count': missing_count,
                'Missing Percentage': f"{missing_percentage:.2f}%"
            })
    
    if missing_info:
        missing_df = pd.DataFrame(missing_info)
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found!")
    
    return missing_info

def handle_missing_data(df):
    """Handle missing values with appropriate strategies."""
    print(f"\n🛠️  HANDLING MISSING DATA:")
    
    df_processed = df.copy()
    
    # Strategy for different types of missing data
    strategies = {}
    
    # Age: Use median (robust to outliers)
    if df_processed['Age'].isnull().sum() > 0:
        age_median = df_processed['Age'].median()
        df_processed['Age'].fillna(age_median, inplace=True)
        strategies['Age'] = f'Filled with median ({age_median:.1f})'
    
    # Cabin: Extract deck information and fill missing
    if df_processed['Cabin'].isnull().sum() > 0:
        # Extract deck (first character) before filling missing values
        df_processed['Deck'] = df_processed['Cabin'].str[0]
        df_processed['Side'] = df_processed['Cabin'].str[-1]
        
        # Fill missing cabins with most common deck
        most_common_deck = df_processed['Deck'].mode()[0]
        most_common_side = df_processed['Side'].mode()[0]
        
        cabin_mask = df_processed['Cabin'].isnull()
        df_processed.loc[cabin_mask, 'Deck'] = most_common_deck
        df_processed.loc[cabin_mask, 'Side'] = most_common_side
        df_processed.loc[cabin_mask, 'Cabin'] = f'{most_common_deck}/0/{most_common_side}'
        
        strategies['Cabin'] = f'Filled with most common pattern ({most_common_deck}/0/{most_common_side})'
    
    # HomePlanet: Use mode (most frequent)
    if df_processed['HomePlanet'].isnull().sum() > 0:
        home_planet_mode = df_processed['HomePlanet'].mode()[0]
        df_processed['HomePlanet'].fillna(home_planet_mode, inplace=True)
        strategies['HomePlanet'] = f'Filled with mode ({home_planet_mode})'
    
    # CryoSleep: Use mode
    if df_processed['CryoSleep'].isnull().sum() > 0:
        cryosleep_mode = df_processed['CryoSleep'].mode()[0]
        df_processed['CryoSleep'].fillna(cryosleep_mode, inplace=True)
        strategies['CryoSleep'] = f'Filled with mode ({cryosleep_mode})'
    
    # Destination: Use mode
    if df_processed['Destination'].isnull().sum() > 0:
        destination_mode = df_processed['Destination'].mode()[0]
        df_processed['Destination'].fillna(destination_mode, inplace=True)
        strategies['Destination'] = f'Filled with mode ({destination_mode})'
    
    # VIP: Use mode
    if df_processed['VIP'].isnull().sum() > 0:
        vip_mode = df_processed['VIP'].mode()[0]
        df_processed['VIP'].fillna(vip_mode, inplace=True)
        strategies['VIP'] = f'Filled with mode ({vip_mode})'
    
    # Amenity spending features: Fill with 0 (logical for missing spending)
    amenity_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for feature in amenity_features:
        if df_processed[feature].isnull().sum() > 0:
            df_processed[feature].fillna(0, inplace=True)
            strategies[feature] = 'Filled with 0 (no spending)'
    
    # Print strategies used
    print("Strategies applied:")
    for col, strategy in strategies.items():
        print(f"  • {col}: {strategy}")
    
    print(f"\n✅ Missing values after processing: {df_processed.isnull().sum().sum()}")
    
    return df_processed

def encode_categorical_features(df):
    """Convert categorical features to numerical format using one-hot encoding."""
    print(f"\n🔄 ENCODING CATEGORICAL FEATURES:")
    
    df_encoded = df.copy()
    
    # Features to one-hot encode
    categorical_to_encode = ['HomePlanet', 'Destination', 'Deck', 'Side']
    
    # Binary features to convert to 0/1
    binary_features = ['CryoSleep', 'VIP']
    
    # One-hot encode categorical features
    for feature in categorical_to_encode:
        if feature in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[feature], prefix=feature, drop_first=False)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(feature, axis=1, inplace=True)
            print(f"  • {feature}: One-hot encoded ({len(dummies.columns)} categories)")
    
    # Convert binary features to 0/1
    for feature in binary_features:
        if feature in df_encoded.columns:
            df_encoded[feature] = df_encoded[feature].astype(int)
            print(f"  • {feature}: Converted to binary (0/1)")
    
    # Drop non-feature columns
    columns_to_drop = ['PassengerId', 'Name', 'Cabin']
    for col in columns_to_drop:
        if col in df_encoded.columns:
            df_encoded.drop(col, axis=1, inplace=True)
    
    return df_encoded

def normalize_numerical_features(df, numerical_features):
    """Normalize numerical features for tanh activation function."""
    print(f"\n📏 NORMALIZING NUMERICAL FEATURES:")
    print("Using StandardScaler (mean=0, std=1) - Optimal for tanh activation")
    print("Rationale:")
    print("  • tanh outputs in range [-1, 1] with steep gradients around 0")
    print("  • Standardized inputs (mean=0, std=1) align with tanh's sensitive region")
    print("  • Prevents saturation and vanishing gradients")
    print("  • Ensures stable and efficient neural network training")
    
    df_normalized = df.copy()
    scaler = StandardScaler()
    
    # Store original statistics for visualization
    original_stats = {}
    normalized_stats = {}
    
    # Normalize numerical features
    features_to_normalize = [col for col in numerical_features if col in df_normalized.columns]
    
    if features_to_normalize:
        # Store original statistics
        for feature in features_to_normalize:
            original_stats[feature] = {
                'mean': df_normalized[feature].mean(),
                'std': df_normalized[feature].std(),
                'min': df_normalized[feature].min(),
                'max': df_normalized[feature].max()
            }
        
        # Apply standardization
        df_normalized[features_to_normalize] = scaler.fit_transform(df_normalized[features_to_normalize])
        
        # Store normalized statistics
        for feature in features_to_normalize:
            normalized_stats[feature] = {
                'mean': df_normalized[feature].mean(),
                'std': df_normalized[feature].std(),
                'min': df_normalized[feature].min(),
                'max': df_normalized[feature].max()
            }
        
        print(f"\n📊 Normalization Summary:")
        for feature in features_to_normalize:
            orig = original_stats[feature]
            norm = normalized_stats[feature]
            print(f"  • {feature}:")
            print(f"    Original: μ={orig['mean']:.2f}, σ={orig['std']:.2f}, range=[{orig['min']:.2f}, {orig['max']:.2f}]")
            print(f"    Normalized: μ={norm['mean']:.3f}, σ={norm['std']:.3f}, range=[{norm['min']:.2f}, {norm['max']:.2f}]")
    
    return df_normalized, scaler, original_stats, normalized_stats

def create_visualizations(df_original, df_processed, numerical_features, original_stats, normalized_stats):
    """Create before/after visualizations of preprocessing effects."""
    print(f"\n📊 CREATING VISUALIZATIONS:")
    
    # Select features for visualization
    viz_features = ['Age', 'FoodCourt']  # Two representative numerical features
    viz_features = [f for f in viz_features if f in numerical_features]
    
    if len(viz_features) < 2:
        # If we don't have both, use whatever numerical features we have
        viz_features = numerical_features[:2]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Preprocessing Effects: Before vs After Normalization', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(viz_features):
        if feature in df_original.columns and feature in df_processed.columns:
            # Before normalization
            axes[i, 0].hist(df_original[feature].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i, 0].set_title(f'{feature} - Before Normalization', fontweight='bold')
            axes[i, 0].set_xlabel(feature)
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Add statistics text
            if feature in original_stats:
                stats = original_stats[feature]
                axes[i, 0].text(0.05, 0.95, f"μ = {stats['mean']:.2f}\nσ = {stats['std']:.2f}", 
                               transform=axes[i, 0].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # After normalization
            axes[i, 1].hist(df_processed[feature].dropna(), bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[i, 1].set_title(f'{feature} - After Normalization', fontweight='bold')
            axes[i, 1].set_xlabel(f'{feature} (Standardized)')
            axes[i, 1].set_ylabel('Frequency')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Add statistics text
            if feature in normalized_stats:
                stats = normalized_stats[feature]
                axes[i, 1].text(0.05, 0.95, f"μ = {stats['mean']:.3f}\nσ = {stats['std']:.3f}", 
                               transform=axes[i, 1].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('preprocessing_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a summary visualization of the tanh function and data distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot tanh function
    x = np.linspace(-4, 4, 1000)
    y = np.tanh(x)
    ax1.plot(x, y, 'b-', linewidth=2, label='tanh(x)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.fill_between(x, y, alpha=0.2)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_xlabel('Input (x)')
    ax1.set_ylabel('tanh(x)')
    ax1.set_title('tanh Activation Function\nMost sensitive around x=0', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot distribution of one normalized feature
    if viz_features and viz_features[0] in df_processed.columns:
        feature = viz_features[0]
        ax2.hist(df_processed[feature].dropna(), bins=50, alpha=0.7, density=True, 
                color='lightgreen', edgecolor='black', label='Normalized Data')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mean = 0')
        ax2.set_xlabel(f'{feature} (Standardized)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Distribution of Normalized {feature}\nCentered around 0 (optimal for tanh)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('tanh_optimization_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the complete analysis and preprocessing pipeline."""
    
    # 1. Load and explore data
    train_df, test_df = load_and_explore_data()
    
    # 2. Describe the dataset
    numerical_features, categorical_features = describe_dataset(train_df)
    
    # 3. Investigate missing values
    missing_info = investigate_missing_values(train_df)
    
    # 4. Handle missing data
    train_processed = handle_missing_data(train_df)
    
    # 5. Encode categorical features
    train_encoded = encode_categorical_features(train_processed)
    
    # 6. Normalize numerical features
    train_final, scaler, original_stats, normalized_stats = normalize_numerical_features(
        train_encoded, numerical_features)
    
    # 7. Create visualizations
    create_visualizations(train_df, train_final, numerical_features, original_stats, normalized_stats)
    
    # Final summary
    print(f"\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"✅ Original shape: {train_df.shape}")
    print(f"✅ Final shape: {train_final.shape}")
    print(f"✅ Missing values: {train_final.isnull().sum().sum()}")
    print(f"✅ Features ready for neural network training with tanh activation")
    
    # Separate features and target
    if 'Transported' in train_final.columns:
        X = train_final.drop('Transported', axis=1)
        y = train_final['Transported'].astype(int)
        print(f"✅ Feature matrix X: {X.shape}")
        print(f"✅ Target vector y: {y.shape}")
        print(f"✅ Target distribution: {y.value_counts().to_dict()}")
    
    return train_final, scaler

if __name__ == "__main__":
    processed_data, scaler = main()
