from compressors import gzip_compressor, bzip2_compressor, lzma_compressor, zstd_compressor, brotli_compressor
from pathlib import Path

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
        raise ValueError(f"Неверное название алгоритма сжатия. Выберите из существующих: {list(compressors_dict.keys())}")
    
    compressor_func = compressors_dict[compressor]

    phi_T1  = compressor_func(train_path, None, lvl)

    if test_path is None:
        return phi_T1
    phi_T1T2 = compressor_func(train_path, test_path, lvl)

    return phi_T1T2 - phi_T1
