import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def main():
    print("Starting model training...")
    
    # Загружаем параметры
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    model_type = params['train']['model_type']
    random_state = params['train']['random_state']
    
    print(f"Parameters: model_type={model_type}, random_state={random_state}")
    
    # Загружаем подготовленные данные
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    print(f"Training data: X_train{X_train.shape}, y_train{y_train.shape}")
    print(f"Test data: X_test{X_test.shape}, y_test{y_test.shape}")
    
    # Настраиваем MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_classification")
    
    with mlflow.start_run():
        # Выбираем модель
        if model_type == "random_forest":
            model = RandomForestClassifier(random_state=random_state, n_estimators=100)
        else:  # logistic_regression по умолчанию
            model = LogisticRegression(random_state=random_state, max_iter=200)
        
        # Обучаем модель
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Логируем в MLflow
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", accuracy)
        
        # Логируем модель как артефакт
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained: {model_type}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Сохраняем модель локально
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        mlflow.log_artifact("model.pkl")
    
    print("Training completed successfully")

if __name__ == "__main__":
    main()