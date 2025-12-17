import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import os

def load_params():
    """Загружаем параметры из params.yaml"""
    with open('params.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_index():
    """Загружаем TF-IDF индекс и метаданные"""
    print("📂 Загрузка поискового индекса...")
    
    # Проверяем существование файлов
    required_files = [
        'data/processed/tfidf_vectorizer.pkl',
        'data/processed/tfidf_matrix.pkl', 
        'data/processed/metadata.csv'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Файл не найден: {file}")
            print("   Сначала запустите: python src/index.py")
            return None, None, None
    
    # Загружаем векторизатор
    with open('data/processed/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Загружаем TF-IDF матрицу
    with open('data/processed/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    
    # Загружаем метаданные
    metadata = pd.read_csv('data/processed/metadata.csv', encoding='utf-8')
    
    print(f"✅ Индекс загружен:")
    print(f"   Документов: {len(metadata)}")
    print(f"   Признаков: {tfidf_matrix.shape[1]}")
    
    return vectorizer, tfidf_matrix, metadata

def clean_query(query: str, params: dict) -> str:
    """Очистка поискового запроса"""
    import re
    
    # Приведение к нижнему регистру
    if params['preprocessing']['lowercase']:
        query = query.lower()
    
    # Удаление пунктуации
    if params['preprocessing']['remove_punctuation']:
        query = re.sub(r'[^\w\s]', ' ', query)
    
    # Удаление чисел
    if params['preprocessing']['remove_numbers']:
        query = re.sub(r'\d+', '', query)
    
    # Удаление лишних пробелов
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query

def search(query: str, vectorizer, tfidf_matrix, metadata, params, top_n=5):
    """Поиск по запросу"""
    # Очищаем запрос
    cleaned_query = clean_query(query, params)
    print(f"🔍 Поиск: '{query}' -> '{cleaned_query}'")
    
    if not cleaned_query:
        print("⚠️  Запрос пустой после очистки")
        return []
    
    # Векторизуем запрос
    query_vector = vectorizer.transform([cleaned_query])
    
    # Вычисляем косинусное сходство
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Сортируем по убыванию сходства
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Фильтруем по порогу
    threshold = params['search']['similarity_threshold']
    top_n = params['search']['top_n']
    
    results = []
    for idx in sorted_indices:
        if similarities[idx] < threshold:
            break
        
        result = {
            'id': int(metadata.iloc[idx]['id']),
            'name': metadata.iloc[idx]['name'],
            'brand': metadata.iloc[idx]['brand'],
            'similarity': float(similarities[idx]),
            'text_preview': metadata.iloc[idx]['cleaned_text'][:100] + '...'
        }
        results.append(result)
        
        if len(results) >= top_n:
            break
    
    return results

def interactive_search():
    """Интерактивный режим поиска"""
    print("🔎 TF-IDF Поисковая система")
    print("=" * 50)
    
    # Загружаем параметры и индекс
    params = load_params()
    vectorizer, tfidf_matrix, metadata = load_index()
    
    if vectorizer is None:
        return
    
    print(f"\n📊 Параметры поиска:")
    print(f"   Кол-во результатов: {params['search']['top_n']}")
    print(f"   Порог сходства: {params['search']['similarity_threshold']}")
    print(f"   Используемые поля: {params['data']['text_columns']}")
    
    print(f"\n💡 Примеры запросов:")
    print("   - 'перчатки нитриловые'")
    print("   - 'электроды экг'") 
    print("   - 'арчдейл' (бренд)")
    print("   - 'гель ультразвуковой'")
    print("\nДля выхода введите 'exit' или 'quit'")
    print("=" * 50)
    
    while True:
        print(f"\n🔍 Введите поисковый запрос: ", end='')
        query = input().strip()
        
        if query.lower() in ['exit', 'quit', 'выход']:
            print("👋 До свидания!")
            break
        
        if not query:
            print("⚠️  Запрос не может быть пустым")
            continue
        
        # Выполняем поиск
        results = search(query, vectorizer, tfidf_matrix, metadata, params)
        
        if not results:
            print(f"❌ По запросу '{query}' ничего не найдено")
            print("   Попробуйте:")
            print("   - Упростить запрос")
            print("   - Использовать другие слова")
            print("   - Проверить орфографию")
        else:
            print(f"\n✅ Найдено результатов: {len(results)}")
            print("-" * 80)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. ID: {result['id']}")
                print(f"   Название: {result['name']}")
                print(f"   Бренд: {result['brand']}")
                print(f"   Сходство: {result['similarity']:.4f}")
                print(f"   Текст: {result['text_preview']}")
                print(f"   {'─' * 60}")

def test_search():
    """Тестовый режим поиска"""
    print("🧪 Тестовый режим поиска")
    print("=" * 50)
    
    # Загружаем параметры и индекс
    params = load_params()
    vectorizer, tfidf_matrix, metadata = load_index()
    
    if vectorizer is None:
        return
    
    # Тестовые запросы
    test_queries = [
        "перчатки нитриловые",
        "электроды экг",
        "ультразвуковой гель",
        "archdale",
        "шприцы инсулиновые"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Тестовый запрос: '{query}'")
        results = search(query, vectorizer, tfidf_matrix, metadata, params)
        
        if results:
            print(f"   ✅ Найдено: {len(results)} результатов")
            for i, result in enumerate(results[:3], 1):  # покажем топ-3
                print(f"   {i}. {result['name'][:50]}... (сходство: {result['similarity']:.4f})")
        else:
            print(f"   ❌ Не найдено")
    
    print(f"\n✅ Тестирование завершено")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_search()
    else:
        interactive_search()
