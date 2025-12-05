### Предполагаемая структура проекта

```tree
.
├── README.md
├── PLAN.md   
├── requirements.txt
│
├── data/
│   ├── raw/{name_author,...}           # Сырые тексты авторов
│   └── processed/{name_author,...}     # Очищенные/нарезанные тексты
│
└── src/
    ├── theory.py          # Реализация метрик/расстояний через сжатие
    ├── data_pipeline.py   # Загрузка/чистка/нарезка текстов
    ├── compressors.py     # Работа с gzip/bzip2/и т.п.
    ├── neural_method.py   # Нейросетевой метод / перплексия
    └── run_experiments.py # Скрипт, который запускает основные эксперименты
```