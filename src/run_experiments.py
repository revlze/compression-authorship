from data_pipeline import process
from pathlib import Path

data_raw = Path('data/raw')

print("==="*30)
print("Генерация датасета")
authors = sorted([author.stem for author in data_raw.glob('*')]) # all authors in `data/raw/`
configs = [
    # Without punctuation, without stop-words, without capital letters
    dict(lowercase=True,  remove_punct=True,  remove_stopwords=True),
    # Without punctuation, with stop-words, without capital letters
    dict(lowercase=True,  remove_punct=True,  remove_stopwords=False),
    # Without punctuation, without stop-words, with capital letters
    dict(lowercase=False, remove_punct=True,  remove_stopwords=True),
    # Without punctuation, with stop-words, with capital letters
    dict(lowercase=False, remove_punct=True,  remove_stopwords=False),
    # With punctuation, without stop-words, without capital letters
    dict(lowercase=True,  remove_punct=False, remove_stopwords=True),
    # With punctuation, without stop-words, with capital letters
    dict(lowercase=False, remove_punct=False, remove_stopwords=True),
    # With punctuation, with stop-words, without capital letters
    dict(lowercase=True,  remove_punct=False, remove_stopwords=False),
    # With punctuation, with stop-words, with capital letters
    dict(lowercase=False, remove_punct=False, remove_stopwords=False),
]

# authors = [authors[0]] # !!! only for test !!! commit this line/ for normal work
for author in authors:
    print(f'Для {author}!')
    for cfg in configs:
        process(
            author,
            train_kb=128,
            test_kb=128,
            seed=2025,
            min_chunk_kb=4,
            **cfg,
        )
    print('==='*30)
        

print("==="*30)
