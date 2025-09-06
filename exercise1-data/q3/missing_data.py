"""
Missing Data Analysis - Spaceship Titanic Dataset
Análise detalhada dos dados faltantes e estratégias de imputação
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.preprocessing import StandardScaler

def analyze_missing_data():
    """
    Analisa dados faltantes do dataset Spaceship Titanic e gera visualização
    """
    
    # Tentar carregar dados reais, senão simular
    try:
        train_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        train_df = simulate_spaceship_data()
    
    # Identificar tipos de features
    numerical_features = []
    categorical_features = []
    
    for col in train_df.columns:
        if col in ['PassengerId', 'Name', 'Transported']:
            continue
        elif train_df[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
        else:
            categorical_features.append(col)
    
    # Análise de missing values
    missing_summary = []
    total_samples = len(train_df)
    
    for col in train_df.columns:
        missing_count = train_df[col].isnull().sum()
        missing_pct = (missing_count / total_samples) * 100
        
        if missing_count > 0:
            missing_summary.append({
                'Feature': col,
                'Missing_Count': missing_count,
                'Missing_Percentage': missing_pct,
                'Type': 'Numerical' if col in numerical_features else 'Categorical'
            })
    
    if not missing_summary:
        return train_df, []
    
    # Criar DataFrame para análise
    missing_df = pd.DataFrame(missing_summary)
    
    # Aplicar imputação
    df_imputed = apply_imputation(train_df, numerical_features, categorical_features)
    
    # Gerar visualização combinada
    fig = create_combined_visualization(train_df, missing_df, df_imputed, numerical_features)
    
    return fig

def simulate_spaceship_data():
    """
    Simula o dataset Spaceship Titanic com missing values realistas
    """
    np.random.seed(42)
    n_samples = 8693  # Tamanho real do dataset
    
    # Simular dados base
    data = {
        'PassengerId': [f'000{i:04d}_01' for i in range(n_samples)],
        'HomePlanet': np.random.choice(['Earth', 'Europa', 'Mars'], n_samples, p=[0.5, 0.3, 0.2]),
        'CryoSleep': np.random.choice([True, False], n_samples, p=[0.36, 0.64]),
        'Cabin': [f'{"ABCDEFG"[np.random.randint(7)]}/{np.random.randint(1, 2000)}/{"PS"[np.random.randint(2)]}' for _ in range(n_samples)],
        'Destination': np.random.choice(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], n_samples, p=[0.4, 0.3, 0.3]),
        'Age': np.random.normal(29, 14, n_samples),
        'VIP': np.random.choice([True, False], n_samples, p=[0.02, 0.98]),
        'RoomService': np.random.exponential(50, n_samples),
        'FoodCourt': np.random.exponential(100, n_samples),
        'ShoppingMall': np.random.exponential(75, n_samples),
        'Spa': np.random.exponential(60, n_samples),
        'VRDeck': np.random.exponential(80, n_samples),
        'Name': [f'Person{i:04d} Surname{i%100:02d}' for i in range(n_samples)],
        'Transported': np.random.choice([True, False], n_samples, p=[0.5, 0.5])
    }
    
    df = pd.DataFrame(data)
    
    # Introduzir missing values com padrões realistas
    # Age: ~15% missing
    age_missing_idx = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
    df.loc[age_missing_idx, 'Age'] = np.nan
    
    # Cabin: ~2% missing  
    cabin_missing_idx = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    df.loc[cabin_missing_idx, 'Cabin'] = np.nan
    
    # Features de gastos: ~10-12% missing cada
    spending_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for feature in spending_features:
        missing_pct = np.random.uniform(0.10, 0.12)  # 10-12%
        missing_idx = np.random.choice(n_samples, size=int(missing_pct * n_samples), replace=False)
        df.loc[missing_idx, feature] = np.nan
    
    # HomePlanet: ~1% missing
    homeplanet_missing_idx = np.random.choice(n_samples, size=int(0.01 * n_samples), replace=False)
    df.loc[homeplanet_missing_idx, 'HomePlanet'] = np.nan
    
    # Destination: ~1% missing
    dest_missing_idx = np.random.choice(n_samples, size=int(0.01 * n_samples), replace=False)
    df.loc[dest_missing_idx, 'Destination'] = np.nan
    
    return df

def apply_imputation(df, numerical_features, categorical_features):
    """
    Aplica estratégias de imputação aos dados
    """
    df_imputed = df.copy()
    
    # Imputação numérica
    for feature in numerical_features:
        if df[feature].isnull().sum() > 0:
            if feature == 'Age':
                impute_value = df[feature].median()
            else:
                impute_value = 0
            df_imputed[feature].fillna(impute_value, inplace=True)
    
    # Imputação categórica
    for feature in categorical_features:
        if df[feature].isnull().sum() > 0:
            mode_val = df[feature].mode()[0] if len(df[feature].mode()) > 0 else 'Unknown'
            df_imputed[feature].fillna(mode_val, inplace=True)
    
    return df_imputed

def create_combined_visualization(df_original, missing_df, df_imputed, numerical_features):
    """
    Cria visualização de dados completos vs dados faltantes
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Análise de Dados Faltantes - Spaceship Titanic', 
                 fontsize=16, fontweight='bold')
    
    # Missing vs Complete data com labels em português
    if not missing_df.empty:
        total_samples = len(df_original)
        missing_counts = [row['Missing_Count'] for _, row in missing_df.iterrows()]
        complete_counts = [total_samples - count for count in missing_counts]
        features = [row['Feature'] for _, row in missing_df.iterrows()]
        
        # Traduzir nomes das features para português
        feature_translation = {
            'Age': 'Idade',
            'Cabin': 'Cabine',
            'HomePlanet': 'Planeta Origem',
            'Destination': 'Destino',
            'RoomService': 'Serviço Quarto',
            'FoodCourt': 'Praça Alimentação',
            'ShoppingMall': 'Shopping',
            'Spa': 'Spa',
            'VRDeck': 'Deck VR'
        }
        
        features_pt = [feature_translation.get(f, f) for f in features]
        
        x_pos = np.arange(len(features))
        bars1 = ax.bar(x_pos, complete_counts, label='Dados Completos', color='lightgreen', alpha=0.8)
        bars2 = ax.bar(x_pos, missing_counts, bottom=complete_counts, label='Dados Faltantes', color='red', alpha=0.8)
        
        ax.set_xlabel('Características', fontsize=14)
        ax.set_ylabel('Número de Amostras', fontsize=14)
        ax.set_title('Dados Completos vs Dados Faltantes', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(features_pt, rotation=45, ha='right', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Adicionar percentuais nos missing data
        for i, (missing_count, total) in enumerate(zip(missing_counts, [total_samples] * len(missing_counts))):
            if missing_count > 0:
                pct = (missing_count / total) * 100
                ax.text(i, complete_counts[i] + missing_count/2, f'{pct:.1f}%', 
                       ha='center', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    return fig


fig = analyze_missing_data()

# Gerar output SVG para mkdocs
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches='tight')
print(buffer.getvalue())

