from src.data_pipeline import process


print("==="*30)
print("Генерация датасета")
authors = [...]

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
for author in authors:
    for cfg in configs:
        process(
            author,
            train_kb=128,
            test_kb=128,
            seed=2025,
            min_chunk_kb=4,
            **cfg,
        )
        

print("==="*30)
