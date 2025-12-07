# stdlib
import gzip
import bz2
import lzma

# внешние
import zstandard as zstd      # pip install zstandard
import brotli                 # pip install brotli



def gzip_compressor(INPUT_PATH: str, lvl: int = 3):

  # Наборы уровней (можете сузить для скорости прогонов)
  # LEVELS = {
  #     "gzip":   [1, 3, 5, 7, 9],        # 1..9
  #     "bzip2":  [1, 3, 5, 7, 9],        # 1..9
  #     "lzma":   [1, 3, 5, 7, 9],        # 1..9
  #     "zstd":   [1, 4, 7, 10, 13, 16, 19, 22],   # 1..22
  #     "brotli": [1, 3, 5, 7, 9, 11],    # 1..11
  # }

  if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Не найден файл: {INPUT_PATH.resolve()}")

  data = INPUT_PATH.read_bytes()

  return gzip.compress(data, compresslevel=lvl)




def bzip2_compressor(INPUT_PATH, lvl: int = 3):

  if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Не найден файл: {INPUT_PATH.resolve()}")

  data = INPUT_PATH.read_bytes()

  return bz2.compress(data, compresslevel=lvl)
    



def lzma_compressor(INPUT_PATH: str, lvl = 3):

  if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Не найден файл: {INPUT_PATH.resolve()}")

  data = INPUT_PATH.read_bytes()

  return lzma.compress(data, compresslevel=lvl)
    




def zstd_compressor(INPUT_PATH: str, lvl: int = 4):

  if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Не найден файл: {INPUT_PATH.resolve()}")

  data = INPUT_PATH.read_bytes()

  cctx = zstd.ZstdCompressor(level=lvl)
  return cctx.compress(data)
    




def brotli_compressor(INPUT_PATH: str, lvl: int = 3):

  if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Не найден файл: {INPUT_PATH.resolve()}")

  data = INPUT_PATH.read_bytes()

  return brotli.compress(data, quality=lvl)



