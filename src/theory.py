from .compressors import gzip_compressor, bzip2_compressor, lzma_compressor, zstd_compressor, brotli_compressor
from .neural_method import llm_compress, paq_compressor, ts_zip_compressor, nncp_compressor, cmix_compressor
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats

def phi(compressor: str, train_path: Path, test_path: Path|None, lvl: int, phi_T1_cache: dict, author: str) -> int:
    '''
    Вычисляем phi(T2/T1) = phi(T1T2) - phi(T1).

    :param compressor: название функции-компрессора
    :param train_path: путь к файлу автора 1 (обучающий текст)
    :param test_path: путь к файлу автора 2 (тестовый слайс)
    :param lvl: уровень сжатия
    :param phi_T1_cache: сохраненные длины обуч. текстов
    :param author: фамилия автора
    :return: значение phi
    '''
    compressors_dict = {
        'gzip': gzip_compressor,
        'bzip2': bzip2_compressor,
        'lzma': lzma_compressor,
        'zstd': zstd_compressor,
        'brotli': brotli_compressor,
        'paq': paq_compressor,
        'llm': llm_compress,
        'ts_zip': ts_zip_compressor,
        'nncp': nncp_compressor,
        'cmix': cmix_compressor,
    }
    
    if compressor not in compressors_dict:
        raise ValueError(f'Неверное название алгоритма сжатия. Выберите из существующих: {list(compressors_dict.keys())}')
    
    compressor_func = compressors_dict[compressor]

    if author in phi_T1_cache:
        phi_T1 = phi_T1_cache[author]
    else:
        phi_T1 = compressor_func(train_path, None, lvl)
        phi_T1_cache[author] = phi_T1

    if test_path is None:
        return phi_T1
    phi_T1T2 = compressor_func(train_path, test_path, lvl)

    return phi_T1T2 - phi_T1


def phi_authors(compressor: str, authors_paths: dict, test_path: Path, lvl: int, phi_T1_cache: dict) -> dict:
    '''
    Для одного слайса вычисляем условное phi со всеми авторами.

    :param compressor: название функции-компрессора
    :param authors_paths: словарь {author_name: path_to_train_text}
    :param test_path: путь к тестовому слайсу
    :param lvl: уровень сжатия
    :param phi_T1_cache: сохраненные длины обуч. текстов
    :return: dict: {author: phi}
    '''
    result = {}
    for author, train_path in authors_paths.items():
        train_path = Path(train_path)
        result[author] = phi(compressor, train_path, test_path, lvl, phi_T1_cache, author)

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
    phi_T1_cache = {}

    for author in authors:
        author_dir = Path('data/processed') / author / preprocess_type
        train_path = author_dir / 'Train.txt'

        if not train_path.exists():
            raise FileNotFoundError(f'Train не найден: {train_path}, проверьте фамилию автора')

        authors_paths[author] = train_path
        phi_T1_cache[author] = phi(compressor, train_path, None, lvl, phi_T1_cache, author)

    table = pd.DataFrame(0, index=authors, columns=authors)

    for author in authors:
        author_dir = Path('data/processed') / author / preprocess_type
        slices_dir = author_dir / f'{n_slices}_slices'

        for i in range(n_slices):
            test_path = slices_dir / f'{i}_slice.txt'
            if not test_path.exists():
                raise FileNotFoundError(f'Слайс {i} у автора {author} не найден: {test_path}')

            phi_vals = phi_authors(compressor, authors_paths, test_path, lvl, phi_T1_cache)

            predicted_author = min(phi_vals, key=phi_vals.get)

            table.loc[author, predicted_author] += 1

    return table


def chi_square(table: pd.DataFrame) -> float:
    """
    Вычисляет хи квадрат для conf-matrix.

    Использует стандартную формулу χ² = Σ (O - E)² / E,
    где O — наблюдаемые значения, E — ожидаемые.

    :param table: DataFrame со значениями a_{i,j}
    :return: chi2_val
    """
    A = table.values.astype(float)
    N = A.sum()

    if N == 0:
        return 0.0

    # Вычисляем суммы по строкам и столбцам
    row_sums = A.sum(axis=1, keepdims=True)
    col_sums = A.sum(axis=0, keepdims=True)

    # Ожидаемые частоты: E_{ij} = (row_sum_i * col_sum_j) / N
    expected = (row_sums @ col_sums) / N

    # Избегаем деления на ноль: учитываем только ячейки с E > 0
    mask = expected > 0
    chi2_val = np.sum(((A - expected) ** 2 / expected)[mask])

    return float(chi2_val)


def significance_level(table: pd.DataFrame) -> tuple[float, float, int]:
    """
    Вычисляет уровень значимости (p-value) для критерия χ².

    H0: все данные подчиняются одному распределению (нет связи автор-предсказание)
    H1: отвергаем H0 (есть статистически значимая связь)

    :param table: DataFrame confusion matrix
    :return: (chi2_val, p_value, degrees_of_freedom)
    """
    n_rows, n_cols = table.shape
    df = (n_rows - 1) * (n_cols - 1)  # степени свободы

    chi2_val = chi_square(table)

    # p-value: вероятность получить χ² >= chi2_val при H0
    p_value = 1 - stats.chi2.cdf(chi2_val, df)

    return chi2_val, p_value, df


def get_accuracy(table: pd.DataFrame) -> float:
    """
    Вычисляет точность классификации (доля правильных предсказаний).

    :param table: DataFrame confusion matrix
    :return: accuracy (0..1)
    """
    correct = table.values.diagonal().sum()
    total = table.values.sum()
    return correct / total if total > 0 else 0.0
