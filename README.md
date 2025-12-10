# Cтруктура проекта

```tree
.
├── README.md
├── PLAN.md   
├── requirements.txt
├── experiments.ipynb      # Ноутбук, в котором запускаются эксперименты
│
├── data/
│   ├── raw/{author_name}/              # Сырые тексты авторов
│   └── processed/
│       └── {author_name}/
│           └── {case}_{punct}_{stop}/  # Варианты предобработки
│               ├── Train.txt
│               ├── Test.txt
│               ├── meta.json
│               └── {num}_slices/       # Нарезанный тест на кусочки
│                   ├── 0_slice.txt
│                   ├── ...
│                   └── n_slice.txt
│
└── src/
    ├── theory.py          # Реализация метрик и построение таблицы
    ├── data_pipeline.py   # Загрузка/обработка/формирование текстов
    ├── compressors.py     # Работа с gzip/bzip2/и т.п.
    ├── neural_method.py   # Нейросетевой метод / перплексия
    └── run_experiments.py # Скрипт, который запускает подготовку данных
```
