# stdlib
import gzip
import bz2
import lzma
from pathlib import Path

# внешние
import zstandard as zstd
import brotli

# Наборы уровней (можете сузить для скорости прогонов)
# LEVELS = {
#     "gzip":   [1, 3, 5, 7, 9],        # 1..9
#     "bzip2":  [1, 3, 5, 7, 9],        # 1..9
#     "lzma":   [1, 3, 5, 7, 9],        # 1..9
#     "zstd":   [1, 4, 7, 10, 13, 16, 19, 22],   # 1..22
#     "brotli": [1, 3, 5, 7, 9, 11],    # 1..11
# }


def _read_file_data(INPUT_PATH: str|Path) -> bytes:
    '''
    Читает содержимое файла как байтовую строку.

    :param INPUT_PATH: Путь к файлу (строка или объект Path)
    :type INPUT_PATH: str | Path
    :return: Содержимое файла в виде байтов
    :rtype: bytes
    :raises FileNotFoundError: Если файл не существует
    '''
    path = Path(INPUT_PATH)

    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path.resolve()}")

    return path.read_bytes()


def _merge(INPUT_PATH_1: str|Path, INPUT_PATH_2: str|Path|None) -> bytes:
    '''
    Читает один или два файла и возвращает их содержимое.

    :param INPUT_PATH_1: Путь к первому файлу (основной текст)
    :type INPUT_PATH_1: str | Path
    :param INPUT_PATH_2: Путь ко второму файлу (тестовый текст) или None.
                        Если None — возвращается только содержимое первого файла.
    :type INPUT_PATH_2: str | Path | None
    :return: Байтовая строка: data1 или data1 + data2
    :rtype: bytes
    '''
    data1 = _read_file_data(INPUT_PATH_1)
    if INPUT_PATH_2 is None:
        return data1
    data2 = _read_file_data(INPUT_PATH_2)
    return data1 + data2


def gzip_compressor(INPUT_PATH_1: str|Path, INPUT_PATH_2: str|Path|None = None, lvl: int = 3) -> int:
    '''
    Сжимает один или два файла с помощью алгоритма gzip и возвращает длину сжатых данных.

    :param INPUT_PATH_1: Путь к первому файлу
    :type INPUT_PATH_1: str | Path
    :param INPUT_PATH_2: Путь ко второму файлу или None. Если указано — файлы объединяются.
    :type INPUT_PATH_2: str | Path | None
    :param lvl: Уровень сжатия (1-9)
    :type lvl: int
    :return: Длина сжатых данных в байтах
    :rtype: int
    '''
    data = _merge(INPUT_PATH_1, INPUT_PATH_2)
    compressed = gzip.compress(data, compresslevel=lvl)

    return len(compressed)


def bzip2_compressor(INPUT_PATH_1: str|Path, INPUT_PATH_2: str|Path|None = None, lvl: int = 3) -> int:
    '''
    Сжимает один или два файла с помощью алгоритма bzip2 и возвращает длину результата.

    :param INPUT_PATH_1: Путь к первому файлу
    :type INPUT_PATH_1: str | Path
    :param INPUT_PATH_2: Путь ко второму файлу или None
    :type INPUT_PATH_2: str | Path | None
    :param lvl: Уровень сжатия (1-9)
    :type lvl: int
    :return: Длина сжатых данных в байтах
    :rtype: int
    '''
    data = _merge(INPUT_PATH_1, INPUT_PATH_2)
    compressed = bz2.compress(data, compresslevel=lvl)

    return len(compressed)
    

def lzma_compressor(INPUT_PATH_1: str|Path, INPUT_PATH_2: str|Path|None = None, lvl: int = 3) -> int:
    '''
    Сжимает один или два файла с помощью алгоритма LZMA и возвращает длину результата.

    :param INPUT_PATH_1: Путь к первому файлу
    :type INPUT_PATH_1: str | Path
    :param INPUT_PATH_2: Путь ко второму файлу или None
    :type INPUT_PATH_2: str | Path | None
    :param lvl: Уровень сжатия (1-9)
    :type lvl: int
    :return: Длина сжатого файла в байтах
    :rtype: int
    '''
    data = _merge(INPUT_PATH_1, INPUT_PATH_2)
    compressed = lzma.compress(data, preset=lvl)

    return len(compressed)
    

def zstd_compressor(INPUT_PATH_1: str|Path, INPUT_PATH_2: str|Path|None = None, lvl: int = 4) -> int:
    '''
    Сжимает один или два файла с помощью Zstandard и возвращает длину результата.

    :param INPUT_PATH_1: Путь к первому файлу
    :type INPUT_PATH_1: str | Path
    :param INPUT_PATH_2: Путь ко второму файлу или None
    :type INPUT_PATH_2: str | Path | None
    :param lvl: Уровень сжатия (1-22)
    :type lvl: int
    :return: Длина сжатых данных в байтах
    :rtype: int
    '''
    data = _merge(INPUT_PATH_1, INPUT_PATH_2)

    cctx = zstd.ZstdCompressor(level=lvl)
    compressed = cctx.compress(data)

    return len(compressed)
    

def brotli_compressor(INPUT_PATH_1: str|Path, INPUT_PATH_2: str|Path|None = None, lvl: int = 3) -> int:
    '''
    Сжимает один или два файла с помощью Brotli и возвращает длину результата.

    :param INPUT_PATH_1: Путь к первому файлу
    :type INPUT_PATH_1: str | Path
    :param INPUT_PATH_2: Путь ко второму файлу или None
    :type INPUT_PATH_2: str | Path | None
    :param lvl: Уровень сжатия (1-11)
    :type lvl: int
    :return: Длина сжатых данных в байтах
    :rtype: int
    '''
    data = _merge(INPUT_PATH_1, INPUT_PATH_2)
    compressed = brotli.compress(data, quality=lvl)

    return len(compressed)
