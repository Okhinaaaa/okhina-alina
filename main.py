import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import random
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if os.uname().sysname == "Darwin":  # Только для macOS
    os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")



SEED = 322
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Градиентный бустинг
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка отображения
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')

# ========== 1. ЗАГРУЗКА ДАННЫХ ==========

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"Размер train: {train.shape}")
print(f"Размер test: {test.shape}")


# Преобразуем дату
train['dt'] = pd.to_datetime(train['dt'])
test['dt'] = pd.to_datetime(test['dt'])

# Сортируем по дате для временных рядов
train = train.sort_values(['product_id', 'dt']).reset_index(drop=True)
test = test.sort_values(['product_id', 'dt']).reset_index(drop=True)

# ========== 2. РАСШИРЕННАЯ ПРЕДОБРАБОТКА ==========
def advanced_preprocess(df, is_train=True, train_stats=None):
    """Расширенная предобработка с генерацией признаков"""
    df = df.copy()
    
    # Календарные признаки
    df['day_of_year'] = df['dt'].dt.dayofyear
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['dt'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['dt'].dt.is_month_end.astype(int)
    df['quarter'] = df['dt'].dt.quarter
    df['week_of_month'] = df['day_of_month'].apply(lambda x: (x-1)//7 + 1)
    
    # Сезонность
    df['season'] = df['month'] % 12 // 3
    season_map = {0: 'winter', 1: 'spring', 2: 'summer', 3: 'autumn'}
    df['season'] = df['season'].map(season_map)
    
    # Взаимодействие признаков
    df['holiday_activity'] = df['holiday_flag'] * df['activity_flag']
    df['weekend_holiday'] = df['is_weekend'] * df['holiday_flag']
    df['weekend_activity'] = df['is_weekend'] * df['activity_flag']
    
    # Кодирование сезона
    season_le = LabelEncoder()
    df['season_encoded'] = season_le.fit_transform(df['season'])
    
    # Признаки иерархии
    df['category_combo_1'] = df['first_category_id'].astype(str) + '_' + df['second_category_id'].astype(str)
    df['category_combo_2'] = df['second_category_id'].astype(str) + '_' + df['third_category_id'].astype(str)
    
    # Обработка пропусков в погодных данных
    weather_cols = ['precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
    
    if is_train:
        # На трейне сохраняем статистики для теста
        weather_stats = {}
        for col in weather_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                weather_stats[col] = median_val
        return df, weather_stats
    else:
        # На тесте используем статистики из трейна
        if train_stats:
            for col in weather_cols:
                if col in df.columns and col in train_stats:
                    df[col] = df[col].fillna(train_stats[col])
        return df

# Применяем предобработку
train, weather_stats = advanced_preprocess(train, is_train=True)
test = advanced_preprocess(test, is_train=False, train_stats=weather_stats)

print("Расширенная предобработка завершена")

# ========== 3. ГЕНЕРАЦИЯ РАСШИРЕННЫХ ПРИЗНАКОВ ==========
def create_advanced_features(df, is_train=True):
    """Создание расширенных признаков с проверкой наличия колонок"""
    df = df.copy()
    
    # Определяем доступные целевые переменные
    available_targets = []
    if is_train:
        if 'price_p05' in df.columns and 'price_p95' in df.columns:
            available_targets = ['price_p05', 'price_p95', 'n_stores']
    else:
        if 'n_stores' in df.columns:
            available_targets = ['n_stores']
    
    windows = [1, 2, 3, 7, 14, 30]
    
    # Создаем лаги и скользящие статистики только для доступных колонок
    for col in available_targets:
        for window in windows:
            df[f'{col}_lag_{window}'] = df.groupby('product_id')[col].shift(window)
            
            df[f'{col}_rolling_mean_{window}'] = df.groupby('product_id')[col]\
                .shift(1).rolling(window=window, min_periods=1).mean()
            df[f'{col}_rolling_std_{window}'] = df.groupby('product_id')[col]\
                .shift(1).rolling(window=window, min_periods=1).std()
            
            if col in ['price_p05', 'price_p95']:
                df[f'{col}_rolling_min_{window}'] = df.groupby('product_id')[col]\
                    .shift(1).rolling(window=window, min_periods=1).min()
                df[f'{col}_rolling_max_{window}'] = df.groupby('product_id')[col]\
                    .shift(1).rolling(window=window, min_periods=1).max()
            
            if window > 1:
                df[f'{col}_diff_{window}'] = df[col] - df.groupby('product_id')[col].shift(window)
                
                if col in ['price_p05', 'price_p95']:
                    df[f'{col}_pct_change_{window}'] = df.groupby('product_id')[col].pct_change(window)
    
    # Взаимодействия лагов с категориями (только если есть соответствующие лаги)
    if is_train:
        for cat_col in ['management_group_id', 'first_category_id']:
            for lag in [1, 3, 7]:
                lag_col_p05 = f'price_p05_lag_{lag}'
                lag_col_p95 = f'price_p95_lag_{lag}'
                
                if lag_col_p05 in df.columns:
                    df[f'price_p05_lag_{lag}_{cat_col}_mean'] = df.groupby(cat_col)[lag_col_p05].transform('mean')
                
                if lag_col_p95 in df.columns:
                    df[f'price_p95_lag_{lag}_{cat_col}_mean'] = df.groupby(cat_col)[lag_col_p95].transform('mean')
    
    # Статистики по категориям (только в train)
    if is_train and 'price_p05' in df.columns:
        df['avg_price_category'] = df.groupby('first_category_id')['price_p05'].transform('mean')
        df['std_price_category'] = df.groupby('first_category_id')['price_p05'].transform('std')
        df['median_price_category'] = df.groupby('first_category_id')['price_p05'].transform('median')
    
    # Признаки активности (если есть соответствующие колонки)
    if 'activity_flag' in df.columns and 'holiday_flag' in df.columns:
        df['activity_holiday_combo'] = df['activity_flag'] * 2 + df['holiday_flag']
        df['weekend_activity'] = df['is_weekend'] * df['activity_flag']
    
    # Сезонность по категориям
    if 'month' in df.columns and 'first_category_id' in df.columns:
        df['month_category'] = df['month'].astype(str) + '_' + df['first_category_id'].astype(str)
    
    # Заполняем пропуски
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if is_train:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)
    
    return df

print("Создание расширенных признаков для train...")
train = create_advanced_features(train, is_train=True)

print("Создание расширенных признаков для test...")
test = create_advanced_features(test, is_train=False)

print(f"Размер train после генерации признаков: {train.shape}")
print(f"Размер test после генерации признаков: {test.shape}")

# ========== 4. ВРЕМЕННАЯ КЛАСТЕРИЗАЦИЯ ТОВАРОВ ==========
def create_temporal_clusters(train_df, test_df, n_clusters=6):
    """Кластеризация товаров по временным паттернам с seed=322"""
    
    print(f"Создание {n_clusters} временных кластеров...")
    
    product_stats_list = []
    
    for product_id in train_df['product_id'].unique():
        product_data = train_df[train_df['product_id'] == product_id]
        
        if len(product_data) > 7:
            stats = {
                'product_id': product_id,
                'price_volatility': product_data['price_p05'].std() if len(product_data) > 1 else 0,
                'price_trend': np.polyfit(range(len(product_data)), product_data['price_p05'].values, 1)[0] if len(product_data) > 1 else 0,
                'avg_price': product_data['price_p05'].mean(),
                'sales_frequency': (product_data['n_stores'] > 0).mean(),
                'weekend_effect': product_data[product_data['dow'].isin([5, 6])]['price_p05'].mean() / 
                                 product_data[~product_data['dow'].isin([5, 6])]['price_p05'].mean() 
                                 if len(product_data[~product_data['dow'].isin([5, 6])]) > 0 else 1,
                'category': product_data['first_category_id'].iloc[0] if len(product_data) > 0 else 0
            }
            product_stats_list.append(stats)
    
    if not product_stats_list:
        print("Недостаточно данных для кластеризации")
        train_df['temporal_cluster'] = 0
        test_df['temporal_cluster'] = 0
        return train_df, test_df
    
    product_stats = pd.DataFrame(product_stats_list)
    
    cluster_features = ['price_volatility', 'price_trend', 'avg_price', 'sales_frequency', 'weekend_effect']
    
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(product_stats[cluster_features].fillna(0))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    product_stats['temporal_cluster'] = kmeans.fit_predict(X_cluster)
    
    print(f"Размеры кластеров:")
    cluster_sizes = product_stats['temporal_cluster'].value_counts().sort_index()
    for cluster, size in cluster_sizes.items():
        print(f"  Кластер {cluster}: {size} товаров")
    
    cluster_map = product_stats.set_index('product_id')['temporal_cluster'].to_dict()
    
    train_df['temporal_cluster'] = train_df['product_id'].map(cluster_map).fillna(-1).astype(int)
    test_df['temporal_cluster'] = test_df['product_id'].map(cluster_map).fillna(-1).astype(int)
    
    for i in range(n_clusters):
        train_df[f'temp_cluster_{i}'] = (train_df['temporal_cluster'] == i).astype(int)
        test_df[f'temp_cluster_{i}'] = (test_df['temporal_cluster'] == i).astype(int)
    
    return train_df, test_df

train, test = create_temporal_clusters(train, test, n_clusters=6)
print(f"Уникальных временных кластеров: {train['temporal_cluster'].nunique()}")

# ========== 5. СНИЖЕНИЕ РАЗМЕРНОСТИ И ДЕТЕКЦИЯ АНОМАЛИЙ ==========
def reduce_weather_dimensionality(train_df, test_df):
    """PCA для погодных данных с seed=322"""
    
    weather_cols = ['precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
    
    available_cols = [col for col in weather_cols if col in train_df.columns]
    
    if len(available_cols) >= 2:
        train_weather = train_df[available_cols].fillna(train_df[available_cols].mean())
        test_weather = test_df[available_cols].fillna(train_df[available_cols].mean())
        
        pca = PCA(n_components=min(2, len(available_cols)), random_state=SEED)
        pca_train = pca.fit_transform(train_weather)
        pca_test = pca.transform(test_weather)
        
        for i in range(pca_train.shape[1]):
            train_df[f'weather_pca_{i+1}'] = pca_train[:, i]
            test_df[f'weather_pca_{i+1}'] = pca_test[:, i]
        
        print(f"Объясненная дисперсия PCA: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        print("Недостаточно погодных данных для PCA")
    
    return train_df, test_df

def detect_anomalies(df):
    """Детекция аномалий в данных с seed=322"""
    
    if 'price_p05' not in df.columns:
        return df
    
    anomaly_features = ['price_p05', 'price_p95', 'n_stores']
    available_features = [f for f in anomaly_features if f in df.columns]
    
    if available_features:
        iso_forest = IsolationForest(contamination=0.05, random_state=SEED, n_jobs=-1)
        X_anomaly = df[available_features].fillna(df[available_features].median())
        df['is_anomaly'] = iso_forest.fit_predict(X_anomaly)
        df['is_anomaly'] = (df['is_anomaly'] == -1).astype(int)
        df['anomaly_flag'] = df['is_anomaly']
        print(f"Найдено аномалий: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
    
    return df

train, test = reduce_weather_dimensionality(train, test)
train = detect_anomalies(train)
test['anomaly_flag'] = 0

# ========== 6. ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ ==========
def prepare_for_training(train_df, test_df):
    """Финальная подготовка данных для моделей с выравниванием признаков"""
    
    exclude_cols = [
        'dt', 'product_id', 'season', 'is_anomaly',
        'category_combo_1', 'category_combo_2', 'month_category',
        'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id'
    ]
    
    target_cols = []
    if 'price_p05' in train_df.columns:
        target_cols = ['price_p05', 'price_p95']
    
    all_features = [col for col in train_df.columns 
                   if col not in exclude_cols + target_cols]
    
    numeric_features = []
    for col in all_features:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            if train_df[col].nunique() > 1:
                numeric_features.append(col)
    
    print(f"\nПризнаков в train: {len(numeric_features)}")
    
    # Убедимся, что в test есть все эти признаки
    missing_in_test = set(numeric_features) - set(test_df.columns)
    if missing_in_test:
        print(f"Добавляем {len(missing_in_test)} недостающих признаков в test...")
        for feature in missing_in_test:
            if feature in train_df.columns:
                test_df[feature] = train_df[feature].median()
            else:
                test_df[feature] = 0
    
    # Заполняем пропуски
    print("\nЗаполнение пропусков...")
    for col in numeric_features:
        if col in train_df.columns:
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
        
        if col in test_df.columns:
            fill_value = train_df[col].median() if col in train_df.columns else 0
            test_df[col] = test_df[col].fillna(fill_value)
    
    # Удаляем из test целевые переменные, если они случайно появились
    for target in ['price_p05', 'price_p95']:
        if target in test_df.columns:
            test_df = test_df.drop(columns=[target])
    
    print(f"\nИтоговые размеры:")
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Количество признаков для обучения: {len(numeric_features)}")
    
    return train_df, test_df, numeric_features

train, test, feature_names = prepare_for_training(train, test)

print(f"\nПервые 10 признаков: {feature_names[:10]}")
print(f"Последние 10 признаков: {feature_names[-10:]}")

# ========== 7. АНСАМБЛИРОВАНИЕ CATBOOST + LIGHTGBM (ФИНАЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ) ==========
def create_model_ensemble(train_df, feature_names, use_gpu=True):
    """Создание ансамбля CatBoost и LightGBM"""
    
    print("="*70)
    print("АНСАМБЛИРОВАНИЕ CATBOOST + LIGHTGBM")
    print("="*70)
    
    X = train_df[feature_names]
    y_p05 = train_df['price_p05']
    y_p95 = train_df['price_p95']
    
    print(f"Размер данных: X={X.shape}, y_p05={y_p05.shape}")
    
    # Разделение для валидации (последние 20%)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_p05_train, y_p05_val = y_p05.iloc[:split_idx], y_p05.iloc[split_idx:]
    y_p95_train, y_p95_val = y_p95.iloc[:split_idx], y_p95.iloc[split_idx:]
    
    print(f"Разделение: Train={len(X_train)}, Val={len(X_val)}")
    
    # 1. LIGHTGBM МОДЕЛИ (НАЧИНАЕМ С НИХ - ОНИ БОЛЕЕ СТАБИЛЬНЫ)
    print("\n" + "-"*50)
    print("ОБУЧЕНИЕ LIGHTGBM МОДЕЛЕЙ")
    print("-"*50)
    
    # Определяем доступное устройство для LightGBM
    try:
        import torch
        lgb_device = 'gpu' if torch.cuda.is_available() and use_gpu else 'cpu'
    except:
        lgb_device = 'cpu'
    
    lgb_params_p05 = {
        'objective': 'quantile',
        'alpha': 0.05,
        'metric': 'quantile',
        'boosting_type': 'gbdt',
        'num_leaves': 45,
        'learning_rate': 0.025,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'min_child_samples': 25,
        'verbose': -1,
        'random_state': SEED,
        'device': lgb_device,
        'n_jobs': -1,
    }
    
    lgb_params_p95 = {
        'objective': 'quantile',
        'alpha': 0.95,
        'metric': 'quantile',
        'boosting_type': 'gbdt',
        'num_leaves': 35,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.75,
        'bagging_freq': 3,
        'min_child_samples': 30,
        'verbose': -1,
        'random_state': SEED,
        'device': lgb_device,
        'n_jobs': -1,
    }
    
    print("Обучение LightGBM для price_p05...")
    train_data_p05 = lgb.Dataset(X_train, label=y_p05_train)
    val_data_p05 = lgb.Dataset(X_val, label=y_p05_val, reference=train_data_p05)
    
    lgb_model_p05 = lgb.train(
        lgb_params_p05,
        train_data_p05,
        valid_sets=[val_data_p05],
        num_boost_round=800,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    print("Обучение LightGBM для price_p95...")
    train_data_p95 = lgb.Dataset(X_train, label=y_p95_train)
    val_data_p95 = lgb.Dataset(X_val, label=y_p95_val, reference=train_data_p95)
    
    lgb_model_p95 = lgb.train(
        lgb_params_p95,
        train_data_p95,
        valid_sets=[val_data_p95],
        num_boost_round=800,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Предсказания LightGBM на валидации
    lgb_pred_p05 = lgb_model_p05.predict(X_val)
    lgb_pred_p95 = lgb_model_p95.predict(X_val)
    
    lgb_mae_p05 = mean_absolute_error(y_p05_val, lgb_pred_p05)
    lgb_mae_p95 = mean_absolute_error(y_p95_val, lgb_pred_p95)
    
    print(f"\nLightGBM метрики на валидации:")
    print(f"  MAE price_p05: {lgb_mae_p05:.4f}")
    print(f"  MAE price_p95: {lgb_mae_p95:.4f}")
    
    # 2. CATBOOST МОДЕЛИ (ПРОБУЕМ, НО НЕ КРИТИЧНО)
    print("\n" + "-"*50)
    print("ОБУЧЕНИЕ CATBOOST МОДЕЛЕЙ (ОПЦИОНАЛЬНО)")
    print("-"*50)
    
    device = 'GPU' if use_gpu else 'CPU'
    print(f"Используется устройство: {device}")
    
    # Безопасные параметры CatBoost (БЕЗ use_best_model!)
    cat_params_safe = {
        'iterations': 400,  # Меньше итераций для скорости
        'depth': 6,
        'learning_rate': 0.05,
        'loss_function': 'Quantile:alpha=0.05',
        'verbose': 100,  # Показываем прогресс
        'random_seed': SEED,
        'task_type': device,
        'use_best_model': False,  # КРИТИЧНО: отключаем!
        'early_stopping_rounds': None,  # Отключаем early stopping
    }
    
    catboost_success = False
    cat_model_p05, cat_model_p95 = None, None
    
    try:
        print("Попытка обучения CatBoost для price_p05...")
        cat_model_p05 = CatBoostRegressor(**cat_params_safe)
        
        # Упрощенное обучение без eval_set (именно это вызывало ошибку)
        cat_model_p05.fit(
            X_train, y_p05_train
        )
        
        print("\nПопытка обучения CatBoost для price_p95...")
        cat_params_p95 = cat_params_safe.copy()
        cat_params_p95['loss_function'] = 'Quantile:alpha=0.95'
        cat_model_p95 = CatBoostRegressor(**cat_params_p95)
        
        cat_model_p95.fit(
            X_train, y_p95_train
        )
        
        # Предсказания CatBoost на валидации
        cat_pred_p05 = cat_model_p05.predict(X_val)
        cat_pred_p95 = cat_model_p95.predict(X_val)
        
        cat_mae_p05 = mean_absolute_error(y_p05_val, cat_pred_p05)
        cat_mae_p95 = mean_absolute_error(y_p95_val, cat_pred_p95)
        
        print(f"\nCatBoost метрики на валидации:")
        print(f"  MAE price_p05: {cat_mae_p05:.4f}")
        print(f"  MAE price_p95: {cat_mae_p95:.4f}")
        
        catboost_success = True
        
    except Exception as e:
        print(f"\n⚠️ CatBoost не удалось обучить: {str(e)}")
        print("Продолжаем только с LightGBM...")
        catboost_success = False
    
    # 3. ОПРЕДЕЛЕНИЕ ВЕСОВ АНСАМБЛЯ
    print("\n" + "-"*50)
    print("ОПРЕДЕЛЕНИЕ ВЕСОВ АНСАМБЛЯ")
    print("-"*50)
    
    if catboost_success:
        # Сравниваем метрики
        lgb_total = lgb_mae_p05 + lgb_mae_p95
        cat_total = cat_mae_p05 + cat_mae_p95
        
        print(f"LightGBM суммарный MAE: {lgb_total:.4f}")
        print(f"CatBoost суммарный MAE: {cat_total:.4f}")
        
        # Определяем веса на основе качества
        if cat_total < lgb_total:
            # CatBoost лучше
            cat_weight = 0.7
            lgb_weight = 0.3
            print(f"CatBoost лучше, используем веса: LGB={lgb_weight:.2f}, Cat={cat_weight:.2f}")
        else:
            # LightGBM лучше или равно
            cat_weight = 0.3
            lgb_weight = 0.7
            print(f"LightGBM лучше или равно, используем веса: LGB={lgb_weight:.2f}, Cat={cat_weight:.2f}")
        
        # Создаем ансамблевые предсказания
        ensemble_pred_p05 = lgb_pred_p05 * lgb_weight + cat_pred_p05 * cat_weight
        ensemble_pred_p95 = lgb_pred_p95 * lgb_weight + cat_pred_p95 * cat_weight
        
        ensemble_mae_p05 = mean_absolute_error(y_p05_val, ensemble_pred_p05)
        ensemble_mae_p95 = mean_absolute_error(y_p95_val, ensemble_pred_p95)
        
        print(f"\nАнсамбль метрики:")
        print(f"  MAE price_p05: {ensemble_mae_p05:.4f} (улучшение: {(lgb_mae_p05 - ensemble_mae_p05)/lgb_mae_p05*100:.1f}%)")
        print(f"  MAE price_p95: {ensemble_mae_p95:.4f} (улучшение: {(lgb_mae_p95 - ensemble_mae_p95)/lgb_mae_p95*100:.1f}%)")
        
        ensemble_weights = (lgb_weight, cat_weight)
    else:
        print("Используем только LightGBM")
        ensemble_weights = (1.0, 0.0)
        ensemble_mae_p05 = lgb_mae_p05
        ensemble_mae_p95 = lgb_mae_p95
    
    # 4. ФИНАЛЬНОЕ ОБУЧЕНИЕ НА ВСЕХ ДАННЫХ
    print("\n" + "-"*50)
    print("ФИНАЛЬНОЕ ОБУЧЕНИЕ НА ВСЕХ ДАННЫХ")
    print("-"*50)
    
    # LightGBM на всех данных
    print("Дообучение LightGBM на всех данных...")
    train_data_p05_full = lgb.Dataset(X, label=y_p05)
    train_data_p95_full = lgb.Dataset(X, label=y_p95)
    
    lgb_final_p05 = lgb.train(
        lgb_params_p05,
        train_data_p05_full,
        num_boost_round=800
    )
    
    lgb_final_p95 = lgb.train(
        lgb_params_p95,
        train_data_p95_full,
        num_boost_round=800
    )
    
    # CatBoost на всех данных (если был успешно обучен)
    if catboost_success:
        print("Дообучение CatBoost на всех данных (без eval_set)...")
        
        # Копируем параметры и УБИРАЕМ use_best_model
        cat_final_params = cat_params_safe.copy()
        cat_final_params['use_best_model'] = False
        cat_final_params['early_stopping_rounds'] = None
        cat_final_params['verbose'] = False  # Отключаем вывод при финальном обучении
        
        try:
            cat_final_p05 = CatBoostRegressor(**cat_final_params)
            cat_final_p05.fit(X, y_p05)
            
            cat_final_params['loss_function'] = 'Quantile:alpha=0.95'
            cat_final_p95 = CatBoostRegressor(**cat_final_params)
            cat_final_p95.fit(X, y_p95)
            
            print("CatBoost успешно дообучен на всех данных")
            
        except Exception as e:
            print(f"⚠️ Не удалось дообучить CatBoost на всех данных: {str(e)}")
            print("Используем только LightGBM...")
            cat_final_p05, cat_final_p95 = None, None
            catboost_success = False
            ensemble_weights = (1.0, 0.0)  # Возвращаемся к чисто LightGBM
    else:
        cat_final_p05, cat_final_p95 = None, None
    
    return {
        'lgb': (lgb_final_p05, lgb_final_p95),
        'cat': (cat_final_p05, cat_final_p95) if catboost_success else None,
        'weights': ensemble_weights,
        'metrics': {
            'lgb_mae': (lgb_mae_p05, lgb_mae_p95),
            'cat_mae': (cat_mae_p05, cat_mae_p95) if catboost_success else None,
            'ensemble_mae': (ensemble_mae_p05, ensemble_mae_p95)
        },
        'catboost_success': catboost_success
    }

# Проверяем доступность GPU
def check_gpu_availability():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

USE_GPU = check_gpu_availability()
print(f"GPU доступен: {USE_GPU}")

# Запускаем ансамблирование
print("\nЗапуск ансамблирования моделей...")
ensemble_result = create_model_ensemble(train, feature_names, use_gpu=USE_GPU)

print("\n" + "="*70)
print("АНСАМБЛИРОВАНИЕ ЗАВЕРШЕНО")
print("="*70)
print(f"Веса ансамбля: LGB={ensemble_result['weights'][0]:.2f}, "
      f"Cat={ensemble_result['weights'][1]:.2f}")
print(f"CatBoost успешен: {ensemble_result['catboost_success']}")

# ========== 8. ПРОСТОЙ ФИНАЛЬНЫЙ САБМИШН БЕЗ ПОСТОБРАБОТКИ ==========
print("\n" + "="*70)
print("СОЗДАНИЕ ПРОСТОГО ФИНАЛЬНОГО САБМИШНА БЕЗ ПОСТОБРАБОТКИ")
print("="*70)

# Подготовка тестовых данных
print("Подготовка тестовых данных...")

X_test = pd.DataFrame(index=test.index)

for feat in feature_names:
    if feat in test.columns:
        X_test[feat] = test[feat]
    else:
        if feat in train.columns:
            X_test[feat] = train[feat].median()
        else:
            X_test[feat] = 0

# Заполняем пропуски
for col in X_test.columns:
    if col in train.columns:
        median_val = train[col].median()
        X_test[col] = X_test[col].fillna(median_val)
    else:
        X_test[col] = X_test[col].fillna(0)

print(f"Размер X_test: {X_test.shape}")

# Создаем предсказания с ансамблированием
print("\nСоздание предсказаний...")

lgb_model_p05, lgb_model_p95 = ensemble_result['lgb']
lgb_weight, cat_weight = ensemble_result['weights']

# Предсказания LightGBM
lgb_pred_p05 = lgb_model_p05.predict(X_test)
lgb_pred_p95 = lgb_model_p95.predict(X_test)

print(f"LightGBM предсказания:")
print(f"  price_p05: [{lgb_pred_p05.min():.3f}, {lgb_pred_p05.max():.3f}]")
print(f"  price_p95: [{lgb_pred_p95.min():.3f}, {lgb_pred_p95.max():.3f}]")

# Если CatBoost доступен, добавляем его предсказания
if ensemble_result['cat'] is not None and cat_weight > 0:
    cat_model_p05, cat_model_p95 = ensemble_result['cat']
    
    cat_pred_p05 = cat_model_p05.predict(X_test)
    cat_pred_p95 = cat_model_p95.predict(X_test)
    
    print(f"\nCatBoost предсказания:")
    print(f"  price_p05: [{cat_pred_p05.min():.3f}, {cat_pred_p05.max():.3f}]")
    print(f"  price_p95: [{cat_pred_p95.min():.3f}, {cat_pred_p95.max():.3f}]")
    
    # Ансамблевые предсказания
    ensemble_pred_p05 = lgb_pred_p05 * lgb_weight + cat_pred_p05 * cat_weight
    ensemble_pred_p95 = lgb_pred_p95 * lgb_weight + cat_pred_p95 * cat_weight
    
    print(f"\nАнсамбль (LGB={lgb_weight:.2f}, Cat={cat_weight:.2f}):")
    print(f"  price_p05 mean: {ensemble_pred_p05.mean():.3f}")
    print(f"  price_p95 mean: {ensemble_pred_p95.mean():.3f}")
    
    final_pred_p05 = ensemble_pred_p05
    final_pred_p95 = ensemble_pred_p95
else:
    print("\nИспользуем только LightGBM (CatBoost недоступен)")
    final_pred_p05 = lgb_pred_p05
    final_pred_p95 = lgb_pred_p95

# МИНИМАЛЬНАЯ ПОСТОБРАБОТКА
print("\nМинимальная проверка корректности...")

# 1. Гарантируем, что p05 <= p95
invalid_pairs = 0
for i in range(len(final_pred_p05)):
    if final_pred_p05[i] > final_pred_p95[i]:
        final_pred_p05[i], final_pred_p95[i] = final_pred_p95[i], final_pred_p05[i]
        invalid_pairs += 1

if invalid_pairs > 0:
    print(f"Исправлено {invalid_pairs} некорректных пар (p05 > p95)")

# 2. Гарантируем неотрицательные цены
neg_p05 = (final_pred_p05 < 0).sum()
neg_p95 = (final_pred_p95 < 0).sum()

if neg_p05 > 0 or neg_p95 > 0:
    print(f"Исправляем отрицательные цены: p05={neg_p05}, p95={neg_p95}")
    final_pred_p05 = np.maximum(final_pred_p05, 0.01)
    final_pred_p95 = np.maximum(final_pred_p95, 0.01)

# 3. Гарантируем минимальный диапазон (0.01)
for i in range(len(final_pred_p05)):
    if final_pred_p95[i] - final_pred_p05[i] < 0.01:
        final_pred_p95[i] = final_pred_p05[i] + 0.01

# Создаем сабмишн
n_predictions = min(len(final_pred_p05), len(test))

submission = pd.DataFrame({
    'row_id': test['row_id'].iloc[:n_predictions],
    'price_p05': final_pred_p05[:n_predictions],
    'price_p95': final_pred_p95[:n_predictions]
})

# Финальная проверка
final_invalid = (submission['price_p05'] > submission['price_p95']).sum()
if final_invalid > 0:
    print(f"ВНИМАНИЕ: Осталось {final_invalid} некорректных пар!")
    mask = submission['price_p05'] > submission['price_p95']
    submission.loc[mask, ['price_p05', 'price_p95']] = submission.loc[mask, ['price_p95', 'price_p05']].values


os.makedirs('results', exist_ok=True)  # Создаем папку если её нет
submission.to_csv('results/submission.csv', index=False)
print("submission.csv сохранен в папку results/")
