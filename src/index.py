import pandas as pd
import yaml
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_params():
    """Загружаем параметры из params.yaml"""
    with open('params.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print("🔧 Построение TF-IDF индекса...")
    
    # Загружаем параметры
    params = load_params()
    print(f"✅ Параметры загружены")
    
    # Загружаем подготовленные данные
    input_path = 'data/processed/prepared_data.csv'
    if not os.path.exists(input_path):
        print(f"❌ Файл {input_path} не найден. Сначала запустите prepare.py")
        return
    
    df = pd.read_csv(input_path, encoding='utf-8')
    print(f"📊 Загружено документов: {len(df)}")
    
    # Подготавливаем тексты
    texts = df['cleaned_text'].tolist()
    print(f"📝 Обработка {len(texts)} текстов...")
    
    # Создаем TF-IDF векторизатор
    print("🏗️  Создаем TF-IDF векторизатор...")
    tfidf_params = params['tfidf']
    
    # Определяем стоп-слова
    stop_words = None
    if tfidf_params['stop_words'] == 'english':
        stop_words = 'english'
    elif tfidf_params['stop_words'] == 'russian':
        # Базовые русские стоп-слова
        stop_words = ['и', 'в', 'на', 'с', 'по', 'для', 'не', 'что', 'это', 'как']
    
    vectorizer = TfidfVectorizer(
        max_features=tfidf_params['max_features'],
        min_df=tfidf_params['min_df'],
        max_df=tfidf_params['max_df'],
        ngram_range=tuple(tfidf_params['ngram_range']),
        stop_words=stop_words
    )
    
    # Обучаем векторизатор и преобразуем тексты
    print("⚙️  Обучение TF-IDF векторизатора...")
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    print(f"✅ TF-IDF матрица создана:")
    print(f"   Размер: {tfidf_matrix.shape[0]} документов × {tfidf_matrix.shape[1]} признаков")
    print(f"   Плотность: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4%}")
    
    # Сохраняем векторизатор и матрицу
    os.makedirs('data/processed', exist_ok=True)
    
    # Сохраняем векторизатор
    vectorizer_path = 'data/processed/tfidf_vectorizer.pkl'
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"💾 Векторизатор сохранен: {vectorizer_path}")
    
    # Сохраняем TF-IDF матрицу
    matrix_path = 'data/processed/tfidf_matrix.pkl'
    with open(matrix_path, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    print(f"💾 TF-IDF матрица сохранена: {matrix_path}")
    
    # Сохраняем метаданные (id документов и исходные тексты)
    metadata = df[['id', 'name', 'brand', 'cleaned_text']].copy()
    metadata_path = 'data/processed/metadata.csv'
    metadata.to_csv(metadata_path, index=False, encoding='utf-8')
    print(f"💾 Метаданные сохранены: {metadata_path}")
    
    # Примеры фич (самые важные слова)
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n📊 Топ-10 самых важных слов (по IDF):")
    
    # IDF - обратная документная частота (чем выше, тем важнее слово)
    idf_values = vectorizer.idf_
    top_indices = np.argsort(idf_values)[-10:]  # 10 самых высоких IDF
    for idx in reversed(top_indices):  # от самого важного к менее важному
        print(f"   {feature_names[idx]}: IDF={idf_values[idx]:.3f}")
    
    # Пример TF-IDF для первого документа
    print(f"\n📝 Пример TF-IDF для первого документа:")
    first_doc_vector = tfidf_matrix[0]
    nonzero_indices = first_doc_vector.nonzero()[1]
    nonzero_values = first_doc_vector.data
    
    # Сортируем по весам
    sorted_indices = np.argsort(nonzero_values)[-5:]  # топ-5 слов
    print(f"   Документ: {texts[0][:80]}...")
    print(f"   Топ-5 слов с наибольшим весом:")
    for idx in sorted_indices:
        word_idx = nonzero_indices[idx]
        weight = nonzero_values[idx]
        print(f"   '{feature_names[word_idx]}': {weight:.4f}")
    
    print(f"\n✅ Индексирование завершено!")
    print(f"📁 Сохраненные файлы:")
    print(f"   - {vectorizer_path} (векторизатор)")
    print(f"   - {matrix_path} (TF-IDF матрица)")
    print(f"   - {metadata_path} (метаданные документов)")

if __name__ == "__main__":
    main()
