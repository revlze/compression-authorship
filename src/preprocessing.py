from data_pipeline import process
from pathlib import Path

def run_data_pipeline(authors: list[str], train_kb: int = 128, test_kb: int = 128, min_chunk_kb: int = 4, seed: int = 2025):
    data_raw = Path('data/raw')

    print("==="*30)
    print("Генерация датасета")
    authors = sorted([(data_raw / author) for author in authors]) # type: ignore
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
        print(f'Для {author}!')
        for cfg in configs:
            process(
                author,
                train_kb=train_kb,
                test_kb=test_kb,
                seed=seed,
                min_chunk_kb=min_chunk_kb,
                **cfg,
            )
        print('==='*30)
            

    print("==="*30)


def split_into_slices(authors: list[str], n_slices: int = 16):
    """
    Делит файл на n равных частей и сохраняет эти части в папку к автору 
    data/processed/{author_name}/{*_*_*}/{count_of_slices}_slices/{num_of_slice}_slice.txt

    :param int n_slices: кол-во сплайнов(слайсов), defaults to 16
    :raises ValueError: n_slices должно быть > 0
    :raises FileNotFoundError: Test.txt у каждого автора должен существовать
    """

    print("==="*30)
    print(f"Начинаю нарезать Test.txt на {n_slices} слайсов, у {authors}!")
    print("==="*30)
    if n_slices <= 0:
        raise ValueError(f"n_slices={n_slices} должно быть > 0")
    
    data_processed = Path('data/processed')

    for author in authors:
        print(f"Автор: {author}")
        available_cfgs_path = list((data_processed/author).glob("*")) 
        print(f'Существуют: {list(map(lambda x: x.stem, available_cfgs_path))}')
        for cfg_path in available_cfgs_path:
            data_slices = cfg_path / f'{n_slices}_slices'
            data_slices.mkdir(parents=True, exist_ok=True)
            test_file = cfg_path / 'Test.txt'
            if not test_file.exists():
                raise FileNotFoundError(f"Нет тестового файла в {cfg_path}, автор: {author}")
            text = test_file.read_text(encoding='utf-8')

            total_len = len(text)
            base_size = total_len // n_slices
            remainder = total_len % n_slices
            start = 0
            for i in range(n_slices):
                size = base_size + (1 if i < remainder else 0)
                slice_file = data_slices / f'{i}_slice.txt'
                slice_file.write_text(text[start:start + size], encoding='utf-8')
                start += size
            print(f"Слайсы сохранены в {data_slices}")
        print("==="*30)

    print("==="*30)

# split_into_slices(['Bunin', 'Bulgakov'], n_slices=2)
