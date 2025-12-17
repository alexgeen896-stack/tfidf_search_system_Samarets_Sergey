import pandas as pd
import os

print("🔍 Анализ датасета offers.csv...")

# Проверяем что файл существует
file_path = 'data/raw/offers.csv'
if not os.path.exists(file_path):
    print(f"❌ Файл {file_path} не найден!")
    print("Проверь путь и попробуй снова.")
    exit(1)

print(f"✅ Файл найден: {file_path}")

try:
    # Пробуем разные кодировки если нужно
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='cp1251')
    
    print(f"\n📊 Датсет успешно загружен")
    print(f"   Размер: {df.shape[0]} строк, {df.shape[1]} колонок")
    
    print(f"\n📝 Колонки ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2}. {col} ({df[col].dtype})")
    
    print(f"\n👀 Первые 3 строки:")
    print(df.head(3).to_string())
    
    print(f"\n🎯 Примеры текстовых полей:")
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_columns[:3]:  # только первые 3 текстовые колонки
        sample = str(df[col].iloc[0])[:150] if len(str(df[col].iloc[0])) > 0 else "ПУСТО"
        print(f"\n   Колонка '{col}':")
        print(f"   Уникальных значений: {df[col].nunique()}")
        print(f"   Пример: {sample}...")
    
    print(f"\n📈 Пропущенные значения:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col, count in missing[missing > 0].items():
            print(f"   {col}: {count} пропусков ({count/len(df)*100:.1f}%)")
    else:
        print("   ✅ Нет пропущенных значений")
    
    print(f"\n💾 Сохраняем информацию о датасете...")
    df.info(verbose=True)
    
except Exception as e:
    print(f"❌ Ошибка при анализе датасета: {e}")
    import traceback
    traceback.print_exc()
