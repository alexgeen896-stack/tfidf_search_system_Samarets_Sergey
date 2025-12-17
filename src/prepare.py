import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def main():
    print("Starting data preparation...")
    
    # Загружаем параметры из params.yaml
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    test_size = params['prepare']['test_size']
    random_state = params['prepare']['random_state']
    
    print(f"Parameters: test_size={test_size}, random_state={random_state}")
    
    # Загружаем данные
    data = pd.read_csv('data/raw/data.csv')
    print(f"Data loaded: {data.shape}")
    
    # Разделяем на features и target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    # Создаем папку если нет
    os.makedirs('data/processed', exist_ok=True)
    
    # Сохраняем обработанные данные
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("Data preparation completed!")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")

if __name__ == "__main__":
    main()