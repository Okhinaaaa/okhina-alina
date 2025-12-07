import pandas as pd
import numpy as np
import re
import gc
import warnings
import random
import os
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')

SEED = 993
random.seed(SEED)
np.random.seed(SEED)


def fast_preprocess(text):
    """Быстрая предобработка текста"""
    if not isinstance(text, str):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def create_fast_features(df):
    """Создание быстрых и эффективных признаков"""
    features = pd.DataFrame(index=df.index)

    features['exact_match'] = df.apply(
        lambda x: 1.0 if x['query_clean'] in x['title_clean'] else 0.0, axis=1
    )

    def word_overlap(row):
        q_words = set(row['query_clean'].split()[:10])
        t_words = set(row['title_clean'].split()[:20])
        if not q_words:
            return 0.0
        return len(q_words & t_words) / len(q_words)
    features['word_overlap'] = df.apply(word_overlap, axis=1)

    features['query_len'] = df['query_clean'].apply(len)
    features['title_len'] = df['title_clean'].apply(len)
    features['desc_len'] = df['desc_clean'].apply(len)

    features['query_words'] = df['query_clean'].apply(lambda x: len(x.split()))
    features['title_words'] = df['title_clean'].apply(lambda x: len(x.split()))

    def jaccard_sim(row):
        q_words = set(row['query_clean'].split())
        t_words = set(row['title_clean'].split())
        if not q_words and not t_words:
            return 0.0
        intersection = len(q_words & t_words)
        union = len(q_words | t_words)
        return intersection / union if union > 0 else 0.0
    features['jaccard'] = df.apply(jaccard_sim, axis=1)

    features['title_starts_with_query'] = df.apply(
        lambda x: 1.0 if x['title_clean'].startswith(x['query_clean'][:20]) else 0.0, axis=1
    )

    for col in ['product_brand', 'product_color', 'product_locale']:
        if col in df.columns:
            le = LabelEncoder()
            features[f'{col}_enc'] = le.fit_transform(df[col].astype(str))

    return features


def create_submission(predictions, test_df):
    """
    Создаёт submission файл в формате:
        id,prediction
    и сохраняет его в results/submission.csv
    """
    test_df = test_df.copy()

    # Гарантируем наличие колонки 'id'
    if 'id' not in test_df.columns:
        test_df['id'] = range(len(test_df))

    submission = pd.DataFrame({
        'id': test_df['id'].values,
        'prediction': predictions
    })

    # Удаляем дубликаты по 'id', если есть
    submission = submission.drop_duplicates(subset=['id']).reset_index(drop=True)

    # Сохраняем
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)

    print(f"✓ Submission успешно сохранён: {submission_path}")
    return submission_path


def main():
    print("=" * 80)
    print("УСКОРЕННЫЙ ПАЙПЛАЙН ДЛЯ РАНЖИРОВАНИЯ (CPU)")
    print("=" * 80)

    # Загрузка данных
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    print(f"\n[1/7] Данные загружены: train={train_df.shape}, test={test_df.shape}")

    # Предобработка текста
    for df in [train_df, test_df]:
        df['query_clean'] = df['query'].fillna('').apply(fast_preprocess)
        df['title_clean'] = df['product_title'].fillna('').apply(fast_preprocess)
        df['desc_clean'] = df['product_description'].fillna('').apply(fast_preprocess)
        for col in ['product_brand', 'product_color', 'product_locale']:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
    print("[2/7] Текст предобработан")

    # Создание признаков
    X_train = create_fast_features(train_df)
    X_test = create_fast_features(test_df)
    print("[3/7] Базовые признаки созданы")

    # TF-IDF + SVD
    all_texts = pd.concat([
        train_df['query_clean'] + ' ' + train_df['title_clean'],
        test_df['query_clean'] + ' ' + test_df['title_clean']
    ])
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    vectorizer.fit(all_texts)

    tfidf_train = vectorizer.transform(train_df['query_clean'] + ' ' + train_df['title_clean'])
    tfidf_test = vectorizer.transform(test_df['query_clean'] + ' ' + test_df['title_clean'])

    svd = TruncatedSVD(n_components=20, random_state=SEED)
    tfidf_svd_train = svd.fit_transform(tfidf_train)
    tfidf_svd_test = svd.transform(tfidf_test)

    for i in range(20):
        X_train[f'tfidf_svd_{i}'] = tfidf_svd_train[:, i]
        X_test[f'tfidf_svd_{i}'] = tfidf_svd_test[:, i]
    print("[4/7] TF-IDF + SVD признаки добавлены")

    # Групповые признаки
    train_group_sizes = train_df.groupby('query_id').size()
    test_group_sizes = test_df.groupby('query_id').size()
    X_train['group_size'] = train_df['query_id'].map(train_group_sizes)
    X_test['group_size'] = test_df['query_id'].map(test_group_sizes)

    for col in ['query_len', 'title_len', 'word_overlap']:
        if col in X_train.columns:
            group_means = train_df.groupby('query_id').apply(lambda x: X_train.loc[x.index, col].mean())
            X_train[f'group_mean_{col}'] = train_df['query_id'].map(group_means)
            X_test[f'group_mean_{col}'] = X_test[col].mean()
    print("[5/7] Групповые признаки добавлены")

    # Подготовка данных
    X_train = X_train.fillna(0).astype('float32')
    X_test = X_test.fillna(0).astype('float32')
    y_train = train_df['relevance'].values
    groups = train_df['query_id'].values
    print(f"[6/7] Обучение модели... (признаков: {X_train.shape[1]})")

    # LightGBM (CPU)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10],
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': 8,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': SEED,
        'n_estimators': 500,
        'verbose': -1,
        # GPU отключён — для совместимости
    }

    gkf = GroupKFold(n_splits=3)
    test_predictions = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups), 1):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
        groups_tr = groups[train_idx]
        groups_val = groups[val_idx]

        train_groups = [np.sum(groups_tr == g) for g in np.unique(groups_tr)]
        val_groups = [np.sum(groups_val == g) for g in np.unique(groups_val)]

        train_data = lgb.Dataset(X_tr, label=y_tr, group=train_groups, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_data, free_raw_data=False)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=params['n_estimators'],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0)
            ]
        )

        test_predictions += model.predict(X_test) / gkf.n_splits
        del model, train_data, val_data
        gc.collect()

    # Постобработка
    final_predictions = test_predictions.copy()

    # Усиление топа
    for query_id in tqdm(test_df['query_id'].unique(), desc="Постобработка: усиление топа"):
        mask = test_df['query_id'] == query_id
        group_preds = final_predictions[mask]
        group_indices = np.where(mask)[0]
        if len(group_preds) <= 1:
            continue
        top_k = min(10, len(group_preds))
        sorted_indices = np.argsort(group_preds)[::-1]
        for rank, idx in enumerate(sorted_indices[:top_k]):
            boost = 1.0 + (top_k - rank) * 0.08
            original_idx = group_indices[idx]
            final_predictions[original_idx] *= boost

    # Ранжирование внутри групп
    for query_id in tqdm(test_df['query_id'].unique(), desc="Постобработка: ранжирование"):
        mask = test_df['query_id'] == query_id
        group_preds = final_predictions[mask]
        if len(group_preds) > 1:
            ranks = pd.Series(group_preds).rank(method='dense', ascending=False)
            denominator = ranks.max() - 1
            if denominator == 0:
                denominator = 1
            normalized = (ranks.max() - ranks) / denominator
            final_predictions[mask] = 0.7 * group_preds + 0.3 * normalized.values

    print("[7/7] Постобработка завершена")

    # Сохранение через обязательную функцию
    create_submission(final_predictions, test_df)


if __name__ == "__main__":
    main()
