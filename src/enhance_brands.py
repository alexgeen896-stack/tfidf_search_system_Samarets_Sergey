
def enhance_prepare_for_brands():
    """Улучшаем prepare.py для лучшего поиска по брендам"""
    # Открываем файл prepare.py
    with open('src/prepare.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Находим место где создается cleaned_text
    # и добавляем сохранение брендов отдельно
    if "df['cleaned_text'] = df['combined_text'].apply" in content:
        # Добавляем код для сохранения брендов
        new_code = '''
    # Сохраняем бренды отдельно для лучшего поиска
    df['brand_cleaned'] = df['brand'].apply(lambda x: clean_text(str(x), params) if pd.notna(x) else "")
    
    # Добавляем бренды к тексту (чтобы они имели больший вес)
    df['enhanced_text'] = df.apply(lambda row: 
        row['brand_cleaned'] + " " + row['brand_cleaned'] + " " + row['cleaned_text'] 
        if row['brand_cleaned'] else row['cleaned_text'], axis=1)
    
    # Используем enhanced_text вместо cleaned_text
    df['final_text'] = df['enhanced_text']
        '''
        
        # Вставляем код после создания cleaned_text
        content = content.replace(
            "df['cleaned_text'] = df['combined_text'].apply(lambda x: clean_text(x, params))",
            "df['cleaned_text'] = df['combined_text'].apply(lambda x: clean_text(x, params))" + new_code
        )
        
        # Обновляем сохранение
        content = content.replace("df['cleaned_text'][:100]", "df['final_text'][:100]")
        
        with open('src/prepare.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ prepare.py обновлен для лучшего поиска по брендам")
    else:
        print("❌ Не удалось обновить prepare.py")

if __name__ == "__main__":
    enhance_prepare_for_brands()
