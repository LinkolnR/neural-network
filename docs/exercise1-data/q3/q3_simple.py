from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Simular dados do Spaceship Titanic para demonstração
np.random.seed(42)

def create_sample_data():
    """Cria dados de amostra completos para demonstração do preprocessing do Spaceship Titanic"""
    n_samples = 1000
    
    # Features numéricas com diferentes escalas
    age = np.random.normal(29, 14, n_samples)
    age = np.clip(age, 0, 80)  # Idades entre 0-80
    
    room_service = np.random.exponential(50, n_samples)
    food_court = np.random.exponential(100, n_samples) 
    spa = np.random.exponential(75, n_samples)
    shopping_mall = np.random.exponential(60, n_samples)
    vr_deck = np.random.exponential(40, n_samples)
    
    # Features categóricas
    home_planets = np.random.choice(['Earth', 'Europa', 'Mars'], n_samples, p=[0.5, 0.3, 0.2])
    destinations = np.random.choice(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], n_samples, p=[0.4, 0.3, 0.3])
    
    # Simulação de cabines (Deck/Num/Side)
    decks = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'], n_samples)
    cabin_nums = np.random.randint(1, 2000, n_samples)
    sides = np.random.choice(['P', 'S'], n_samples)
    cabins = [f"{deck}/{num}/{side}" for deck, num, side in zip(decks, cabin_nums, sides)]
    
    # Features booleanas
    cryo_sleep = np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    vip = np.random.choice([True, False], n_samples, p=[0.1, 0.9])
    
    # Introduzir missing values conforme percentuais da seção 3.2
    age[np.random.choice(n_samples, size=int(0.15*n_samples), replace=False)] = np.nan
    room_service[np.random.choice(n_samples, size=int(0.104*n_samples), replace=False)] = np.nan
    food_court[np.random.choice(n_samples, size=int(0.102*n_samples), replace=False)] = np.nan
    shopping_mall[np.random.choice(n_samples, size=int(0.107*n_samples), replace=False)] = np.nan
    spa[np.random.choice(n_samples, size=int(0.116*n_samples), replace=False)] = np.nan
    vr_deck[np.random.choice(n_samples, size=int(0.112*n_samples), replace=False)] = np.nan
    
    # Missing values para categóricas
    home_planets_missing = np.random.choice(n_samples, size=int(0.011*n_samples), replace=False)
    for idx in home_planets_missing:
        home_planets[idx] = None
        
    destinations_missing = np.random.choice(n_samples, size=int(0.011*n_samples), replace=False)
    for idx in destinations_missing:
        destinations[idx] = None
        
    cabins_missing = np.random.choice(n_samples, size=int(0.019*n_samples), replace=False)
    for idx in cabins_missing:
        cabins[idx] = None
    
    # Criar DataFrame
    df = pd.DataFrame({
        'HomePlanet': home_planets,
        'CryoSleep': cryo_sleep,
        'Cabin': cabins,
        'Destination': destinations,
        'Age': age,
        'VIP': vip,
        'RoomService': room_service,
        'FoodCourt': food_court,
        'ShoppingMall': shopping_mall,
        'Spa': spa,
        'VRDeck': vr_deck
    })
    
    return df

def apply_preprocessing_strategies(df):
    """
    Aplica todas as estratégias de pré-processamento conforme seção 3.2:
    1. Tratamento de dados faltantes
    2. Encoding de variáveis categóricas  
    3. Normalização Z-score otimizada para tanh
    """
    df_processed = df.copy()
    
    print("=== ESTRATÉGIA 1: TRATAMENTO DE DADOS FALTANTES ===")
    print("\nDados faltantes ANTES do tratamento:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        pct = (missing / len(df)) * 100
        if missing > 0:
            print(f"  {col}: {missing} ({pct:.1f}%)")
    
    # 1.1 Features Numéricas - conforme estratégias da seção 3.2
    print("\n1.1 Tratamento de Features Numéricas:")
    
    # Age: Imputação pela mediana (preserva distribuição)
    if 'Age' in df_processed.columns:
        median_age = df_processed['Age'].median()
        df_processed['Age'].fillna(median_age, inplace=True)
        print(f"  • Age: preenchido com mediana = {median_age:.1f}")
    
    # Features de gastos: Preenchimento com zero (ausência = não utilizou)
    expense_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for feature in expense_features:
        if feature in df_processed.columns:
            df_processed[feature].fillna(0, inplace=True)
            print(f"  • {feature}: preenchido com 0 (não utilizou serviço)")
    
    # 1.2 Features Categóricas - Imputação pela moda
    print("\n1.2 Tratamento de Features Categóricas:")
    categorical_features = ['HomePlanet', 'Cabin', 'Destination']
    for feature in categorical_features:
        if feature in df_processed.columns:
            mode_value = df_processed[feature].mode()[0] if not df_processed[feature].mode().empty else 'Unknown'
            df_processed[feature].fillna(mode_value, inplace=True)
            print(f"  • {feature}: preenchido com moda = '{mode_value}'")
    
    print("\n=== ESTRATÉGIA 2: ENCODING DE VARIÁVEIS CATEGÓRICAS ===")
    
    # 2.1 One-Hot Encoding para categóricas nominais
    print("\n2.1 One-Hot Encoding:")
    if 'HomePlanet' in df_processed.columns:
        planet_dummies = pd.get_dummies(df_processed['HomePlanet'], prefix='HomePlanet')
        df_processed = pd.concat([df_processed, planet_dummies], axis=1)
        df_processed.drop('HomePlanet', axis=1, inplace=True)
        print(f"  • HomePlanet → {len(planet_dummies.columns)} colunas binárias")
    
    if 'Destination' in df_processed.columns:
        dest_dummies = pd.get_dummies(df_processed['Destination'], prefix='Destination')
        df_processed = pd.concat([df_processed, dest_dummies], axis=1)
        df_processed.drop('Destination', axis=1, inplace=True)
        print(f"  • Destination → {len(dest_dummies.columns)} colunas binárias")
    
    # Cabin: extrair Deck e Side, depois one-hot
    if 'Cabin' in df_processed.columns:
        # Extrair Deck e Side da cabin (formato: Deck/Num/Side)
        cabin_parts = df_processed['Cabin'].str.split('/', expand=True)
        if len(cabin_parts.columns) >= 3:
            df_processed['Deck'] = cabin_parts[0]
            df_processed['Side'] = cabin_parts[2]
            
            deck_dummies = pd.get_dummies(df_processed['Deck'], prefix='Deck')
            side_dummies = pd.get_dummies(df_processed['Side'], prefix='Side')
            
            df_processed = pd.concat([df_processed, deck_dummies, side_dummies], axis=1)
            df_processed.drop(['Cabin', 'Deck', 'Side'], axis=1, inplace=True)
            print(f"  • Cabin → Deck ({len(deck_dummies.columns)}) + Side ({len(side_dummies.columns)}) colunas")
    
    # 2.2 Label Encoding para booleanas
    print("\n2.2 Label Encoding para Booleanas:")
    boolean_features = ['CryoSleep', 'VIP']
    for feature in boolean_features:
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].astype(int)
            print(f"  • {feature}: False→0, True→1")
    
    print("\n=== ESTRATÉGIA 3: NORMALIZAÇÃO Z-SCORE OTIMIZADA PARA TANH ===")
    
    # Identificar apenas features numéricas para normalização
    numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n3.1 Features numéricas identificadas para normalização: {len(numeric_features)}")
    for feature in numeric_features:
        mean_val = df_processed[feature].mean()
        std_val = df_processed[feature].std()
        print(f"  • {feature}: μ={mean_val:.2f}, σ={std_val:.2f}")
    
    # Aplicar Z-score normalization
    scaler = StandardScaler()
    df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])
    
    print(f"\n3.2 Z-score aplicado: X_norm = (X - μ) / σ")
    print("  ✓ Dados centralizados em μ=0, σ=1")
    print("  ✓ Região ótima para tanh: [-2, +2]")
    print("  ✓ ~68% dos dados na região ativa [-1, +1]")
    
    return df_processed, scaler

def demonstrate_preprocessing():
    """Demonstra o impacto completo do preprocessing para redes neurais"""
    
    # Criar dados de demonstração
    print("Criando dados de amostra do Spaceship Titanic...")
    df_original = create_sample_data()
    
    print(f"\nDataset original: {df_original.shape}")
    print(f"Features: {list(df_original.columns)}")
    
    # Aplicar estratégias de preprocessing
    df_processed, scaler = apply_preprocessing_strategies(df_original)
    
    print(f"\nDataset após preprocessing: {df_processed.shape}")
    print(f"Novas features: {list(df_processed.columns)}")
    
    # Verificar se não há mais missing values
    missing_after = df_processed.isnull().sum().sum()
    print(f"\nMissing values após preprocessing: {missing_after}")
    
    return df_original, df_processed, scaler

def create_comprehensive_visualization():
    """Cria visualização completa do impacto do preprocessing"""
    
    # Executar preprocessing
    df_original, df_processed, scaler = demonstrate_preprocessing()
    
    # Selecionar features numéricas originais para comparação
    numeric_original = ['Age', 'RoomService', 'FoodCourt', 'Spa']
    
    # Filtrar apenas features que existem no dataset processado
    available_features = [f for f in numeric_original if f in df_processed.columns]
    
    if len(available_features) < 2:
        print("Aviso: Poucas features numéricas disponíveis para visualização")
        return None
    
    # Criar subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Impacto Completo do Preprocessing para Redes Neurais\n(Dataset Spaceship Titanic - Estratégias Seção 3.2)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plotar as duas primeiras features disponíveis
    for i, feature in enumerate(available_features[:2]):
        # ANTES: dados originais com missing values
        original_data = df_original[feature].dropna()
        axes[i, 0].hist(original_data, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[i, 0].set_title(f'{feature} - ANTES do Preprocessing\n(com missing values)')
        axes[i, 0].set_xlabel(feature)
        axes[i, 0].set_ylabel('Frequência')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Estatísticas originais
        mean_orig = original_data.mean()
        std_orig = original_data.std()
        axes[i, 0].axvline(mean_orig, color='red', linestyle='--', linewidth=2, 
                          label=f'μ={mean_orig:.1f}, σ={std_orig:.1f}')
        axes[i, 0].legend()
        
        # Adicionar info sobre missing values
        missing_count = df_original[feature].isnull().sum()
        missing_pct = (missing_count / len(df_original)) * 100
        axes[i, 0].text(0.05, 0.95, f'Missing: {missing_count} ({missing_pct:.1f}%)', 
                       transform=axes[i, 0].transAxes, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                       verticalalignment='top')
        
        # DEPOIS: dados normalizados (Z-score)
        processed_data = df_processed[feature]
        axes[i, 1].hist(processed_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[i, 1].set_title(f'{feature} - DEPOIS do Preprocessing\n(Z-score normalizado)')
        axes[i, 1].set_xlabel(f'{feature} (μ=0, σ=1)')
        axes[i, 1].set_ylabel('Frequência')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Destacar região ótima para tanh
        axes[i, 1].axvspan(-1, 1, alpha=0.2, color='green', label='Região Ótima\npara tanh (~68%)')
        axes[i, 1].axvspan(-2, 2, alpha=0.1, color='orange', label='Região Ativa\npara tanh (~95%)')
        axes[i, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='μ=0')
        axes[i, 1].legend()
        
        # Verificar estatísticas após normalização
        mean_norm = processed_data.mean()
        std_norm = processed_data.std()
        axes[i, 1].text(0.05, 0.95, f'μ={mean_norm:.3f}, σ={std_norm:.3f}', 
                       transform=axes[i, 1].transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                       verticalalignment='top')
    
    plt.tight_layout()
    
    # Adicionar texto explicativo detalhado
    textstr = '''ESTRATÉGIAS DE PREPROCESSING IMPLEMENTADAS (Seção 3.2):

1. TRATAMENTO DE MISSING VALUES:
   • Age: Mediana (preserva distribuição)
   • Gastos: Zero (ausência = não utilizou)
   • Categóricas: Moda (preserva proporções)

2. ENCODING CATEGÓRICO:
   • One-Hot: HomePlanet, Destination, Cabin
   • Label: CryoSleep, VIP (booleanas)

3. NORMALIZAÇÃO Z-SCORE:
   • X_norm = (X - μ) / σ
   • Otimizada para função tanh
   • Região ativa: [-2, +2], ótima: [-1, +1]'''
    
    fig.text(0.02, 0.02, textstr, fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.9))
    
    return fig

# Executar demonstração completa e gerar SVG
if __name__ == "__main__":
    fig = create_comprehensive_visualization()
    if fig:
        buffer = StringIO()
        plt.savefig(buffer, format="svg", transparent=True, bbox_inches='tight', dpi=300)
        print(buffer.getvalue())
    else:
        print("Erro na geração da visualização")
