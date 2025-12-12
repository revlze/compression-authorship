"""
Нейросетевые методы сжатия для определения авторства.

Реализованы компрессоры на основе нейросетей с тем же интерфейсом,
что и в compressors.py: принимают пути к файлам, возвращают длину сжатых данных.
"""

from pathlib import Path
from typing import Any

# Импортируем вспомогательные функции из compressors
from .compressors import _read_file_data, _merge

# Глобальный кэш для моделей и токенизаторов
# Структура: {model_name: (tokenizer, model)}
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def clear_model_cache() -> None:
    """
    Очищает кэш загруженных моделей для освобождения памяти.
    """
    global _MODEL_CACHE
    _MODEL_CACHE.clear()


def _get_cached_model(model_name: str, model_class: type, tokenizer_class: type) -> tuple[Any, Any]:
    """
    Получает модель и токенизатор из кэша или загружает их.

    :param model_name: Название модели (например, 'gpt2' или 'ai-forever/rugpt3small_based_on_gpt2')
    :param model_class: Класс модели (например, GPT2LMHeadModel или AutoModelForCausalLM)
    :param tokenizer_class: Класс токенизатора (например, GPT2Tokenizer или AutoTokenizer)
    :return: Кортеж (tokenizer, model)
    """
    global _MODEL_CACHE

    if model_name not in _MODEL_CACHE:
        tokenizer = tokenizer_class.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)

    return _MODEL_CACHE[model_name]


def nncp_compressor(INPUT_PATH_1: str|Path, INPUT_PATH_2: str|Path|None = None, lvl: int = 3) -> int:
    """
    Сжимает данные с помощью NNCP (Neural Network Compression).

    NNCP использует Transformer для предсказания следующего байта и
    арифметическое кодирование на основе предсказанных вероятностей.

    Примечание: NNCP — это внешняя утилита (не pip-пакет), требуется бинарник в PATH.
    См. https://bellard.org/nncp/

    :param INPUT_PATH_1: Путь к первому файлу
    :param INPUT_PATH_2: Путь ко второму файлу или None
    :param lvl: Уровень сжатия (размер модели): 1-small, 2-medium, 3-large
    :return: Длина сжатых данных в байтах
    """
    import subprocess
    import tempfile
    import os
    import shutil

    if not shutil.which('nncp'):
        raise RuntimeError(
            "NNCP не найден в PATH. Скачайте бинарник с https://bellard.org/nncp/\n"
            "и добавьте в PATH."
        )

    data = _merge(INPUT_PATH_1, INPUT_PATH_2)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_in:
        tmp_in.write(data)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path + '.nncp'

    try:
        # nncp c <input> <output> — сжатие
        subprocess.run(
            ['nncp', 'c', tmp_in_path, tmp_out_path],
            check=True,
            capture_output=True
        )
        compressed_size = os.path.getsize(tmp_out_path)
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)

    return compressed_size


def pac_compressor(INPUT_PATH_1: str|Path, INPUT_PATH_2: str|Path|None = None, lvl: int = 3) -> int:
    """
    Сжимает данные с помощью PAC (Prediction by Partial Matching with neural net).

    Использует нейросеть для контекстного моделирования.

    :param INPUT_PATH_1: Путь к первому файлу
    :param INPUT_PATH_2: Путь ко второму файлу или None
    :param lvl: Уровень (не используется, для совместимости)
    :return: Длина сжатых данных в байтах
    """
    try:
        import paq
    except ImportError:
        raise ImportError("PAQ не установлен. Попробуйте: pip install paq")

    data = _merge(INPUT_PATH_1, INPUT_PATH_2)
    compressed = paq.compress(data)

    return len(compressed)


def cmix_compressor(INPUT_PATH_1: str|Path, INPUT_PATH_2: str|Path|None = None, lvl: int = 3) -> int:
    """
    Сжимает данные с помощью cmix — одного из лучших компрессоров,
    использующего нейросети для контекстного смешивания.

    Требует установленный бинарник cmix в PATH.

    :param INPUT_PATH_1: Путь к первому файлу
    :param INPUT_PATH_2: Путь ко второму файлу или None
    :param lvl: Уровень сжатия (не используется cmix)
    :return: Длина сжатых данных в байтах
    """
    import subprocess
    import tempfile
    import os

    data = _merge(INPUT_PATH_1, INPUT_PATH_2)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_in:
        tmp_in.write(data)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path + '.cmix'

    try:
        subprocess.run(
            ['cmix', '-c', tmp_in_path, tmp_out_path],
            check=True,
            capture_output=True
        )
        compressed_size = os.path.getsize(tmp_out_path)
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)

    return compressed_size


def ts_zip_compressor(INPUT_PATH_1: str|Path, INPUT_PATH_2: str|Path|None = None, lvl: int = 3) -> int:
    """
    Сжимает данные с помощью ts_zip — компрессора на основе GPT-2.

    Использует предобученную языковую модель для арифметического кодирования.
    Модель кэшируется для ускорения повторных вызовов.

    :param INPUT_PATH_1: Путь к первому файлу
    :param INPUT_PATH_2: Путь ко второму файлу или None
    :param lvl: Уровень (1=gpt2, 2=gpt2-medium, 3=gpt2-large)
    :return: Длина сжатых данных в байтах
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch
    except ImportError:
        raise ImportError(
            "Требуются зависимости: pip install transformers torch"
        )
    import math

    model_names = {1: 'gpt2', 2: 'gpt2-medium', 3: 'gpt2-large'}
    model_name = model_names.get(lvl, 'gpt2')

    data = _merge(INPUT_PATH_1, INPUT_PATH_2)
    text = data.decode('utf-8', errors='replace')

    # Используем кэшированную модель
    tokenizer, model = _get_cached_model(model_name, GPT2LMHeadModel, GPT2Tokenizer)

    tokens = tokenizer.encode(text, return_tensors='pt')

    # Вычисляем log-вероятности для арифметического кодирования
    total_bits = 0.0
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits

        for i in range(tokens.shape[1] - 1):
            probs = torch.softmax(logits[0, i], dim=-1)
            actual_token = tokens[0, i + 1]
            prob = probs[actual_token].item()
            if prob > 0:
                total_bits += -math.log2(prob)

    # Возвращаем размер в байтах
    compressed_size = int(total_bits / 8) + 1

    return compressed_size


def llm_compress(INPUT_PATH_1: str|Path, INPUT_PATH_2: str|Path|None = None, lvl: int = 3) -> int:
    """
    Сжатие на основе LLM (языковой модели) через вычисление кросс-энтропии.

    Размер сжатых данных = количество бит, необходимых для кодирования
    текста с использованием распределения вероятностей от LLM.
    Модель кэшируется для ускорения повторных вызовов.

    :param INPUT_PATH_1: Путь к первому файлу
    :param INPUT_PATH_2: Путь ко второму файлу или None
    :param lvl: Уровень (размер модели): 1=small, 2=medium, 3=large
    :return: Эквивалентная длина сжатых данных в байтах
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        import math
    except ImportError:
        raise ImportError("Требуются зависимости: pip install transformers torch")

    # Для русского текста лучше использовать ruGPT, для английского — GPT2
    model_names = {
        1: 'ai-forever/rugpt3small_based_on_gpt2',
        2: 'ai-forever/rugpt3medium_based_on_gpt2',
        3: 'ai-forever/rugpt3large_based_on_gpt2'
    }
    model_name = model_names.get(lvl, model_names[1])

    data = _merge(INPUT_PATH_1, INPUT_PATH_2)
    text = data.decode('utf-8', errors='replace')

    # Используем кэшированную модель
    tokenizer, model = _get_cached_model(model_name, AutoModelForCausalLM, AutoTokenizer)

    # Разбиваем на чанки если текст слишком длинный
    max_length = model.config.n_positions if hasattr(model.config, 'n_positions') else 1024

    tokens = tokenizer.encode(text, return_tensors='pt', truncation=False)

    total_bits = 0.0

    with torch.no_grad():
        # Обрабатываем по частям
        for start in range(0, tokens.shape[1] - 1, max_length):
            end = min(start + max_length, tokens.shape[1])
            chunk = tokens[:, start:end]

            outputs = model(chunk)
            logits = outputs.logits

            for i in range(chunk.shape[1] - 1):
                probs = torch.softmax(logits[0, i], dim=-1)
                actual_token = chunk[0, i + 1]
                prob = probs[actual_token].item()
                if prob > 0:
                    total_bits += -math.log2(prob)

    compressed_size = int(total_bits / 8) + 1

    return compressed_size