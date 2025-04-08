#!/usr/bin/env python
import argparse
import os
from dataclasses import dataclass, asdict
from pathlib import Path

# Загрузка переменных окружения из .env файла
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from catboost import CatBoostClassifier
from clearml import Task

@dataclass
class Config:
    seed: int = 2024
    test_size: float = 0.2
    iterations: int = 500
    verbose: int = False
    
    # Параметры CatBoost
    depth: int = 4
    learning_rate: float = 0.06
    loss_function: str = "MultiClass"
    custom_metric: list = None
    colsample_bylevel: float = 0.098
    subsample: float = 0.95
    l2_leaf_reg: int = 9
    min_data_in_leaf: int = 243
    max_bin: int = 187
    random_strength: int = 1
    task_type: str = "CPU"
    thread_count: int = -1
    bootstrap_type: str = "Bernoulli"
    early_stopping_rounds: int = 50
    
    def __post_init__(self):
        if self.custom_metric is None:
            self.custom_metric = ["Recall"]

def seed_everything(seed=2024):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_data():
    url = 'https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/quickstart_train.csv'
    rides_info = pd.read_csv(url)
    return rides_info

def preprocess_data(rides_info):
    cat_features = ["model", "car_type", "fuel_type"]
    targets = ["target_class", "target_reg"]
    features2drop = ["car_id"]
    
    filtered_features = [i for i in rides_info.columns if (i not in targets and i not in features2drop)]
    num_features = [i for i in filtered_features if i not in cat_features]
    
    for c in cat_features:
        rides_info[c] = rides_info[c].astype(str)
    
    return rides_info, filtered_features, cat_features, num_features, targets

def perform_eda(rides_info, task):
    # Создаем графики для EDA и логируем их в ClearML
    
    # 1. Распределение целевой переменной
    plt.figure(figsize=(10, 6))
    sns.countplot(x='target_class', data=rides_info)
    plt.title('Распределение классов')
    plt.xticks(rotation=45)
    plt.tight_layout()
    task.logger.report_matplotlib_figure(
        title='Распределение классов',
        series='',
        figure=plt.gcf()
    )
    plt.close()
    
    # 2. Корреляционная матрица числовых признаков
    numeric_data = rides_info.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    correlation = numeric_data.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                linewidths=0.5, cbar_kws={'shrink': .8})
    plt.title('Корреляционная матрица')
    plt.tight_layout()
    task.logger.report_matplotlib_figure(
        title='Корреляционная матрица',
        series='',
        figure=plt.gcf()
    )
    plt.close()
    
    # 3. Распределение по типам автомобилей
    plt.figure(figsize=(10, 6))
    sns.countplot(x='car_type', data=rides_info)
    plt.title('Распределение по типам автомобилей')
    plt.tight_layout()
    task.logger.report_matplotlib_figure(
        title='Распределение по типам автомобилей',
        series='',
        figure=plt.gcf()
    )
    plt.close()
    
    # 4. Распределение по типам топлива
    plt.figure(figsize=(10, 6))
    sns.countplot(x='fuel_type', data=rides_info)
    plt.title('Распределение по типам топлива')
    plt.tight_layout()
    task.logger.report_matplotlib_figure(
        title='Распределение по типам топлива',
        series='',
        figure=plt.gcf()
    )
    plt.close()
    
    # 5. Боксплот рейтингов автомобилей по типам
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='car_type', y='car_rating', data=rides_info)
    plt.title('Рейтинги автомобилей по типам')
    plt.tight_layout()
    task.logger.report_matplotlib_figure(
        title='Рейтинги автомобилей по типам',
        series='',
        figure=plt.gcf()
    )
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500, help="Number of boosting iterations")
    parser.add_argument("--verbose", type=int, default=False, help="Enable CatBoost training progress output (0 for False, any other number for verbose frequency)")
    args = parser.parse_args()
    
    # Инициализация конфигурации
    cfg = Config(iterations=args.iterations, verbose=args.verbose)
    
    # Инициализация ClearML
    task = Task.init(project_name="CatBoost Classification", 
                    task_name="Car Bugs Classification",
                    tags=["catboost", "classification", "hard-mode"])
    
    # Логирование гиперпараметров
    task.connect(asdict(cfg))
    
    # Установка сида для воспроизводимости
    seed_everything(cfg.seed)
    
    # Загрузка данных
    rides_info = load_data()
    
    # Предобработка данных
    rides_info, filtered_features, cat_features, num_features, targets = preprocess_data(rides_info)
    
    # Логирование информации о данных
    task.logger.report_table(
        title="Data Sample",
        series="Sample",
        table_plot=rides_info.head()
    )
    
    # Проведение EDA и логирование графиков
    perform_eda(rides_info, task)
    
    # Разделение на тренировочную и валидационную выборки
    train, test = train_test_split(rides_info, test_size=cfg.test_size, random_state=cfg.seed)
    
    X_train = train[filtered_features]
    y_train = train["target_class"]
    
    X_test = test[filtered_features]
    y_test = test["target_class"]
    
    # Логирование размеров выборок
    task.logger.report_text(
        f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}"
    )
    
    # Настройка параметров CatBoost
    cb_params = {
        "depth": cfg.depth,
        "learning_rate": cfg.learning_rate,
        "loss_function": cfg.loss_function,
        "custom_metric": cfg.custom_metric,
        "cat_features": cat_features,
        "colsample_bylevel": cfg.colsample_bylevel,
        "subsample": cfg.subsample,
        "l2_leaf_reg": cfg.l2_leaf_reg,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "max_bin": cfg.max_bin,
        "random_strength": cfg.random_strength,
        "task_type": cfg.task_type,
        "thread_count": cfg.thread_count,
        "bootstrap_type": cfg.bootstrap_type,
        "random_seed": cfg.seed,
        "early_stopping_rounds": cfg.early_stopping_rounds,
        "iterations": cfg.iterations
    }
    
    # Инициализация и обучение модели
    model = CatBoostClassifier(**cb_params)
    
    # Обучение модели с логированием в ClearML
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=cfg.verbose
    )
    
    # Логирование процесса обучения происходит автоматически через патч ClearML
    
    # Сохранение модели
    model_path = "cb_model.cbm"
    model.save_model(model_path)
    task.upload_artifact("model", artifact_object=model_path)
    
    # Расчет и логирование метрик
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    task.logger.report_scalar(
        title="Metrics",
        series="Accuracy",
        value=accuracy,
        iteration=0
    )
    
    # Classification Report
    cls_report = classification_report(y_test, y_pred, target_names=y_test.unique(), output_dict=True)
    cls_report_df = pd.DataFrame(cls_report).T
    
    # Логирование classification report
    task.logger.report_table(
        title="Classification Report",
        series="Report",
        table_plot=cls_report_df
    )
    
    # Логирование важности признаков
    feature_importance = pd.DataFrame({
        'Feature': model.feature_names_,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Важность признаков')
    plt.tight_layout()
    task.logger.report_matplotlib_figure(
        title='Важность признаков',
        series='',
        figure=plt.gcf()
    )
    plt.close()
    
    print(f"Обучение завершено. Accuracy: {accuracy:.4f}")
    print(f"Модель сохранена в {model_path}")
    print("Результаты доступны в ClearML")
    
    # Улучшенное решение для завершения ClearML с учетом возможного медленного интернета
    print("Завершение задачи ClearML...")
    task.mark_completed()

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
