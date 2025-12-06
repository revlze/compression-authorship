### Предполагаемая структура проекта

```tree
.
├── README.md
├── PLAN.md   
├── requirements.txt
│
├── data/
│   ├── raw/{author_name}/              # Сырые тексты авторов
│   └── processed/
│       └── {author_name}/
│           └── {case}_{punct}_{stop}/  # Варианты предобработки
│               ├── Train.txt
│               ├── Test.txt
│               └── meta.json
│
└── src/
    ├── theory.py          # Реализация метрик/расстояний через сжатие
    ├── data_pipeline.py   # Загрузка/чистка/нарезка текстов
    ├── compressors.py     # Работа с gzip/bzip2/и т.п.
    ├── neural_method.py   # Нейросетевой метод / перплексия
    └── run_experiments.py # Скрипт, который запускает основные эксперименты
```