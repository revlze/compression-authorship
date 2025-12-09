from data_pipeline import process
from pathlib import Path, PurePath

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
                train_kb=128,
                test_kb=128,
                seed=2025,
                min_chunk_kb=4,
                **cfg,
            )
        print('==='*30)
            

    print("==="*30)


def split_into_splines(authors: list[str], n_splines: int = 16):
    """
    Делит файл на n равных частей и сохраняет эти части в папку к автору 
    data/processed/{author_name}/{*_*_*}/{count_of_splines}_splines/{num_of_spline}_spline.txt

    :param int n_splines: кол-во сплайнов(слайсов), defaults to 16
    :param int | None n_first_authors: кол-во авторов, которых возьмет алгоритм на обработку, defaults to None
    :raises ValueError: n_splines должно быть > 0
    :raises FileNotFoundError: Test.txt у каждого автора должен существовать
    """

    print("==="*30)
    print(f"Начинаю нарезать Test.txt на {n_splines} слайсов, у {authors}!")
    print("==="*30)
    if n_splines <= 0:
        raise ValueError(f"n_splines={n_splines} должно быть > 0")
    
    data_processed = Path('data/processed')

    for author in authors:
        print(f"Автор: {author}")
        available_cfgs_path = list((data_processed/author).glob("*")) 
        print(f'Существуют: {list(map(lambda x: x.stem, available_cfgs_path))}')
        for cfg_path in available_cfgs_path:
            data_splines = cfg_path / f'{n_splines}_splines'
            data_splines.mkdir(parents=True, exist_ok=True)
            text = ''
            test_file = cfg_path / 'Test.txt'
            if test_file.exists():
                text = test_file.read_text(encoding='utf-8')
            else:
                raise FileNotFoundError(f"Нет тестового файла в {cfg_path}, автор: {author}")

            total_len = len(text)
            base_size = total_len // n_splines
            remainder = total_len % n_splines
            start = 0
            for i in range(n_splines):
                size = base_size + (1 if i < remainder else 0)
                spline_file = data_splines / f'{i}_spline.txt'
                spline_file.write_text(text[start:start + size])
                start += size
            print(f"Слайсы сохранены в {data_splines}")
        print("==="*30)

    print("==="*30)

# split_into_splines(['Bunin', 'Bulgakov'], n_splines=2)
