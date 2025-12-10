from compressors import gzip_compressor, bzip2_compressor, lzma_compressor, zstd_compressor, brotli_compressor
from pathlib import Path
import pandas as pd

def phi(compressor: str, train_path: Path, test_path: Path|None, lvl: int) -> int:
    '''
    Вычисляем phi(T2/T1) = phi(T1T2) - phi(T1).

    :param compressor: название функции-компрессора
    :param train_path: путь к файлу автора 1 (обучающий текст)
    :param test_path: путь к файлу автора 2 (тестовый слайс)
    :param lvl: уровень сжатия
    :return: значение phi
    '''
    compressors_dict = {
        'gzip': gzip_compressor,
        'bzip2': bzip2_compressor,
        'lzma': lzma_compressor,
        'zstd': zstd_compressor,
        'brotli': brotli_compressor
    }
    
    if compressor not in compressors_dict:
        raise ValueError(f'Неверное название алгоритма сжатия. Выберите из существующих: {list(compressors_dict.keys())}')
    
    compressor_func = compressors_dict[compressor]

    phi_T1  = compressor_func(train_path, None, lvl)

    if test_path is None:
        return phi_T1
    phi_T1T2 = compressor_func(train_path, test_path, lvl)

    return phi_T1T2 - phi_T1


def phi_authors(compressor: str, authors_paths: dict, test_path: Path, lvl: int) -> dict:
    '''
    Для одного слайса вычисляем условное phi со всеми авторами.

    :param compressor: название функции-компрессора
    :param authors_paths: словарь {author_name: path_to_train_text}
    :param test_path: путь к тестовому слайсу
    :param lvl: уровень сжатия
    :return: dict: {author: phi}
    '''
    result = {}
    for author, train_path in authors_paths.items():
        train_path = Path(train_path)
        result[author] = phi(compressor, train_path, test_path, lvl)

    return result


def build_table(authors: list[str],  preprocess_type: str, n_slices: int, compressor: str, lvl: int) -> pd.DataFrame:
    '''
    Строим таблицу conf-matrix для всех авторов: сколько раз каждый автор
    был определён как ближайший по phi.

    :param authors: список фамилий авторов
    :param preprocess_type: тип предобработки, например origcase_withpunct_withstop
    :param n_slices: количество тестовых слайсов
    :param compressor: название функции-компрессора
    :param lvl: уровень сжатия
    :return: pd.DataFrame, таблица ближайших авторов
    '''
    authors_paths = {}
    for author in authors:
        author_dir = Path('data/processed') / author / preprocess_type
        train_path = author_dir / 'Train.txt'

        if not train_path.exists():
            raise FileNotFoundError(f'Train не найден: {train_path}, проверьте фамилию автора')

        authors_paths[author] = train_path

    table = pd.DataFrame(0, index=authors, columns=authors)

    for author in authors:
        author_dir = Path('data/processed') / author / preprocess_type
        slices_dir = author_dir / f'{n_slices}_slices'

        for i in range(n_slices):
            test_path = slices_dir / f'{i}_slice.txt'
            if not test_path.exists():
                raise FileNotFoundError(f'Слайс {i} у автора {author} не найден: {test_path}')

            phi_vals = phi_authors(compressor, authors_paths, test_path, lvl)

            predicted_author = min(phi_vals, key=phi_vals.get)

            table.loc[author, predicted_author] += 1

    return table


def chi_square(table: pd.DataFrame):
    """
    Вычисляет хи квадрат для conf-matrix.

    :param table: DataFrame со значениями a_{i,j}
    :return: chi2_val
    """
    A = table.values.astype(float)
    N = A.sum()

    P = A / N

    p_i = P.sum(axis=1).reshape(-1, 1)
    p_j = P.sum(axis=0).reshape(1, -1)

    expected = N * (p_i @ p_j)

    chi2_val = ((A - expected)**2 / expected).sum()

    return chi2_val
