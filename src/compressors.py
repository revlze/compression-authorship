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



def gzip_compressor(INPUT_PATH: str|Path, lvl: int = 3) -> bytes:
  '''
  Сжимает файл с помощью алгоритма gzip.
  
  :param INPUT_PATH: Путь к файлу для сжатия
  :type INPUT_PATH: str | Path
  :param lvl: Уровень сжатия (1-9, по умолчанию 3)
  :type lvl: int
  :return: Сжатые данные в виде байтов
  :rtype: bytes
  '''
  data = _read_file_data(INPUT_PATH)

  return gzip.compress(data, compresslevel=lvl)




def bzip2_compressor(INPUT_PATH: str|Path, lvl: int = 3) -> bytes:
  '''
  Сжимает файл с помощью алгоритма bzip2.
  
  :param INPUT_PATH: Путь к файлу для сжатия
  :type INPUT_PATH: str | Path
  :param lvl: Уровень сжатия (1-9, по умолчанию 3)
  :type lvl: int
  :return: Сжатые данные в виде байтов
  :rtype: bytes
  ''' 
  data = _read_file_data(INPUT_PATH)

  return bz2.compress(data, compresslevel=lvl)
    



def lzma_compressor(INPUT_PATH: str|Path, lvl: int = 3) -> bytes:
  '''
  Сжимает файл с помощью алгоритма LZMA.
  
  :param INPUT_PATH: Путь к файлу для сжатия
  :type INPUT_PATH: str | Path
  :param lvl: Уровень сжатия (1-9, по умолчанию 3)
  :type lvl: int
  :return: Сжатые данные в виде байтов
  :rtype: bytes
  '''
  data = _read_file_data(INPUT_PATH)

  return lzma.compress(data, preset=lvl)
    




def zstd_compressor(INPUT_PATH: str|Path, lvl: int = 4) -> bytes:
  '''
  Сжимает файл с помощью алгоритма Zstandard.
  
  :param INPUT_PATH: Путь к файлу для сжатия
  :type INPUT_PATH: str | Path
  :param lvl: Уровень сжатия (1-22, по умолчанию 4)
  :type lvl: int
  :return: Сжатые данные в виде байтов
  :rtype: bytes
  '''
  data = _read_file_data(INPUT_PATH)

  cctx = zstd.ZstdCompressor(level=lvl)
  return cctx.compress(data)
    




def brotli_compressor(INPUT_PATH: str|Path, lvl: int = 3) -> bytes:
  '''
  Сжимает файл с помощью алгоритма Brotli.
  
  :param INPUT_PATH: Путь к файлу для сжатия
  :type INPUT_PATH: str | Path
  :param lvl: Уровень сжатия (1-11, по умолчанию 3)
  :type lvl: int
  :return: Сжатые данные в виде байтов
  :rtype: bytes
  '''
  data = _read_file_data(INPUT_PATH)

  return brotli.compress(data, quality=lvl)



