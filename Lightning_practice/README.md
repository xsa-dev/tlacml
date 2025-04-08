# Tourch Lighting and Clear ML tools

Библиотека для работы с PyTorch и MLOps, включающая:

- Утилиты для работы с Lightning
- Кастомные слои и модели
- Инструменты для визуализации и анализа моделей

## Установка

### Требования

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (для запуска через Makefile)

### Загрузка данных

```bash
make download_data
```

### Проверка

```bash
make dev
```

### Обучение

```bash
make start
```

## Структура проекта

```bash
├── light/               # Основной пакет с модулями
│   ├── __init__.py      # Инициализация пакета
│   ├── data.py          # Модуль для работы с данными
│   └── module.py        # Модуль с моделью нейронной сети
├── main.py              # Точка входа в приложение
├── pyproject.toml       # Конфигурация зависимостей
└── Makefile             # Команды для управления проектом
```

## Архитектура и workflow

Проект реализует классификатор языка жестов на основе Sign MNIST датасета:

### Data Pipeline (`light.data.SignMNISTDataModule`)

- Загрузка CSV с изображениями 28x28 пикселей
- Нормализация и аугментация данных:

```python
transforms.Normalize((0.5,), (0.5,)),
transforms.RandomHorizontalFlip(p=0.1),
transforms.RandomRotation(degrees=(-180, 180))
```

- Разделение на train/val/test DataLoader'ы

### Модель (`light.module.SignLanguageModel`)

- CNN архитектура:

```python
self.block1 = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.BatchNorm2d(8),
    nn.AvgPool2d(2),
    nn.ReLU()
)
```

- Метрики: accuracy, loss
- Оптимизатор: Adam с lr=1e-3

### Процесс обучения

#### Конфигурация

- Настройка через `Trainer`

```python
Trainer(
    accelerator="auto",
    max_epochs=10,
    callbacks=[ModelCheckpoint(monitor="train_loss")]
)
```

#### Особенности

- Логирование метрик
- Сохранение лучших весов
- Автоматическое определение доступного ускорителя (CPU/GPU/Metal)

Для запуска полного цикла используйте:

```bash
make download_data  # загрузить датасет
make start          # обучение с сохранением моделей
```

## Лицензия

MIT
