import pandas as pd
import yaml
import os
import re
from typing import List, Optional
import sys

def load_params():
    """Загружаем параметры из params.yaml"""
    try:
        with open('params.yaml', 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        
        # Проверяем обязательные ключи
        required_keys = ['data', 'tfidf', 'preprocessing', 'search']
        for key in required_keys:
            if key not in params:
                print(f"❌ В params.yaml отсутствует ключ: {key}")
                sys.exit(1)
        
        return params
    except Exception as e:
        print(f"❌ Ошибка при загрузке params.yaml: {e}")
        sys.exit(1)

def clean_text(text: str, params: dict, is_brand: bool = False) -> str:
    """Очистка текста"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Приведение к нижнему регистру
    if params['preprocessing']['lowercase']:
        text = text.lower()
    
    # Для брендов не удаляем пунктуацию полностью
    if params['preprocessing']['remove_punctuation'] and not is_brand:
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # Удаление чисел
    if params['preprocessing']['remove_numbers'] and not is_brand:
        text = re.sub(r'\d+', '', text)
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def combine_text_columns(row: pd.Series, text_columns: List[str], params: dict) -> str:
    """Объединяет несколько текстовых колонок в одну строку с учетом брендов"""
    texts = []
    brand_text = ""
    
    for col in text_columns:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            text = str(row[col]).strip()
            
            # Особая обработка для брендов
            if col == 'brand':
                brand_text = clean_text(text, params, is_brand=True)
                # Добавляем бренд несколько раз для увеличения веса
                if brand_text:
                    texts.append(brand_text)
                    texts.append(brand_text)  # дублируем для веса
            else:
                texts.append(clean_text(text, params, is_brand=False))
    
    # Добавляем бренд еще раз в конце для гарантии
    if brand_text:
        texts.append(brand_text)
    
    return " ".join(texts)

def main():
    print("🚀 Подготовка данных для TF-IDF системы...")
    
    # Загружаем параметры
    params = load_params()
    print(f"✅ Параметры загружены")
    print(f"   Файл: {params['data']['input_file']}")
    print(f"   Текстовые колонки: {params['data']['text_columns']}")
    
    try:
        # Загружаем данные с разными кодировками
        file_path = params['data']['input_file']
        for encoding in ['utf-8', 'cp1251', 'latin1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✅ Данные загружены с кодировкой {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            print("❌ Не удалось определить кодировку файла")
            sys.exit(1)
            
        print(f"📊 Данные загружены: {df.shape[0]} строк, {df.shape[1]} колонок")
        
        # Проверяем существование колонок
        missing_cols = []
        for col in params['data']['text_columns']:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"⚠️  Отсутствующие колонки: {missing_cols}")
            print(f"   Доступные колонки: {list(df.columns)}")
            # Используем только существующие колонки
            text_columns = [col for col in params['data']['text_columns'] if col in df.columns]
            print(f"   Будем использовать: {text_columns}")
        else:
            text_columns = params['data']['text_columns']
        
        # Объединяем текстовые колонки с усилением брендов
        print(f"🔧 Объединяем текстовые колонки (с усилением брендов)...")
        df['combined_text'] = df.apply(
            lambda row: combine_text_columns(row, text_columns, params), 
            axis=1
        )
        
        # Очищаем текст окончательно
        print("🧹 Финальная очистка текста...")
        df['cleaned_text'] = df['combined_text'].apply(lambda x: clean_text(x, params, is_brand=False))
        
        # Удаляем пустые тексты
        initial_count = len(df)
        df = df[df['cleaned_text'].str.strip() != ''].copy()
        removed = initial_count - len(df)
        print(f"📉 Удалено пустых текстов: {removed}")
        
        # Сохраняем подготовленные данные
        os.makedirs('data/processed', exist_ok=True)
        output_path = 'data/processed/prepared_data.csv'
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"✅ Подготовленные данные сохранены: {output_path}")
        print(f"📈 Статистика:")
        print(f"   - Исходных записей: {initial_count}")
        print(f"   - После очистки: {len(df)}")
        print(f"   - Использовано колонок: {len(text_columns)}")
        
        # Примеры очищенных текстов
        print(f"\n📝 Примеры очищенных текстов:")
        samples = min(3, len(df))
        for i in range(samples):
            original = df.iloc[i]['combined_text']
            cleaned = df.iloc[i]['cleaned_text']
            print(f"\n   Запись {i+1}:")
            print(f"   Оригинал: {original[:80]}..." if len(original) > 80 else f"   Оригинал: {original}")
            print(f"   Очищенный: {cleaned[:80]}..." if len(cleaned) > 80 else f"   Очищенный: {cleaned}")
        
        # Статистика по брендам
        if 'brand' in df.columns:
            brand_stats = df['brand'].dropna().value_counts()
            print(f"\n🏷️  Статистика по брендам (топ-5):")
            for brand, count in brand_stats.head().items():
                print(f"   {brand}: {count} товаров")
        
    except Exception as e:
        print(f"❌ Ошибка при обработке данных: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
