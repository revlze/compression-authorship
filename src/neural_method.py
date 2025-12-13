"""
Нейросетевые методы сжатия для определения авторства.

Реализованы компрессоры на основе нейросетей с тем же интерфейсом,
что и в compressors.py: принимают пути к файлам, возвращают длину сжатых данных.
"""

import math
from pathlib import Path
from typing import Any

# Импортируем вспомогательные функции из compressors
from .compressors import _read_file_data, _merge

# Глобальный кэш для моделей и токенизаторов
# Структура: {(model_name, device, dtype): (tokenizer, model)}
_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}

# Таймаут для subprocess (секунды)
_SUBPROCESS_TIMEOUT = 300


def clear_model_cache() -> None:
    """
    Очищает кэш загруженных моделей для освобождения памяти.
    """
    global _MODEL_CACHE
    _MODEL_CACHE.clear()


def _get_device() -> str:
    """Автоопределение лучшего доступного устройства."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _get_dtype(device: str) -> "torch.dtype":
    """Выбор оптимального dtype для устройства."""
    import torch
    if device == "cpu":
        return torch.float32
    # Prefer bfloat16 if available (better numerical stability)
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _get_cached_model(
    model_name: str,
    model_class: type,
    tokenizer_class: type,
    device: str | None = None,
    dtype: "torch.dtype | None" = None,
) -> tuple[Any, Any]:
    """
    Получает модель и токенизатор из кэша или загружает их.

    :param model_name: Название модели
    :param model_class: Класс модели
    :param tokenizer_class: Класс токенизатора
    :param device: Устройство (cuda/mps/cpu), None = auto
    :param dtype: Тип данных, None = auto
    :return: Кортеж (tokenizer, model)
    """
    import torch
    global _MODEL_CACHE

    if device is None:
        device = _get_device()
    if dtype is None:
        dtype = _get_dtype(device)

    cache_key = (model_name, device, str(dtype))

    if cache_key not in _MODEL_CACHE:
        tokenizer = tokenizer_class.from_pretrained(model_name)
        # Ensure pad_token is set for proper loss computation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = model_class.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        model.to(device)
        model.eval()
        _MODEL_CACHE[cache_key] = (tokenizer, model)

    return _MODEL_CACHE[cache_key]


def _get_max_length(model) -> int:
    """Получить максимальную длину контекста модели."""
    config = model.config
    # Try different config attributes (varies by model)
    for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
        if hasattr(config, attr):
            val = getattr(config, attr)
            if val is not None:
                return val
    return 1024  # fallback


def _run_subprocess(cmd: list[str], timeout: int = _SUBPROCESS_TIMEOUT) -> "subprocess.CompletedProcess":
    """
    Запуск subprocess с таймаутом и человекочитаемыми ошибками.
    """
    import subprocess

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=timeout,
        )
        return result
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Команда превысила таймаут ({timeout}s): {' '.join(cmd)}\n"
            f"stdout: {e.stdout.decode('utf-8', errors='replace') if e.stdout else ''}\n"
            f"stderr: {e.stderr.decode('utf-8', errors='replace') if e.stderr else ''}"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Команда завершилась с ошибкой (код {e.returncode}): {' '.join(cmd)}\n"
            f"stdout: {e.stdout.decode('utf-8', errors='replace') if e.stdout else ''}\n"
            f"stderr: {e.stderr.decode('utf-8', errors='replace') if e.stderr else ''}"
        ) from e


def _decode_text(data: bytes, strict: bool = False) -> str:
    """
    Декодирует bytes в текст.

    :param data: Входные данные
    :param strict: Если True, падает на не-UTF8. Если False, заменяет невалидные символы.
    :raises ValueError: Если strict=True и данные не UTF-8
    """
    if strict:
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Данные не являются валидным UTF-8 (позиция {e.start}). "
                "Используйте strict=False для замены невалидных символов."
            ) from e
    return data.decode('utf-8', errors='replace')


def nncp_compressor(INPUT_PATH_1: str | Path, INPUT_PATH_2: str | Path | None = None, lvl: int = 3) -> int:
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
    import tempfile
    import shutil

    if not shutil.which('nncp'):
        raise RuntimeError(
            "NNCP не найден в PATH. Скачайте бинарник с https://bellard.org/nncp/\n"
            "и добавьте в PATH."
        )

    # Маппинг уровня на размер модели NNCP
    model_sizes = {1: 's', 2: 'm', 3: 'l'}
    model_flag = model_sizes.get(lvl, 'l')

    data = _merge(INPUT_PATH_1, INPUT_PATH_2)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_in:
        tmp_in.write(data)
        tmp_in_path = Path(tmp_in.name)

    tmp_out_path = tmp_in_path.with_suffix('.bin.nncp')

    try:
        # nncp c -m <model> <input> <output> — сжатие
        _run_subprocess(['nncp', 'c', '-m', model_flag, str(tmp_in_path), str(tmp_out_path)])
        compressed_size = tmp_out_path.stat().st_size
    finally:
        if tmp_in_path.exists():
            tmp_in_path.unlink()
        if tmp_out_path.exists():
            tmp_out_path.unlink()

    return compressed_size


def paq_compressor(INPUT_PATH_1: str | Path, INPUT_PATH_2: str | Path | None = None, lvl: int = 3) -> int:
    """
    Сжимает данные с помощью PAQ (Prediction by Partial Matching).

    PAQ использует контекстное моделирование и арифметическое кодирование.

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


def cmix_compressor(INPUT_PATH_1: str | Path, INPUT_PATH_2: str | Path | None = None, lvl: int = 3) -> int:
    """
    Сжимает данные с помощью cmix — одного из лучших компрессоров,
    использующего нейросети для контекстного смешивания.

    Требует установленный бинарник cmix в PATH.

    :param INPUT_PATH_1: Путь к первому файлу
    :param INPUT_PATH_2: Путь ко второму файлу или None
    :param lvl: Уровень сжатия (не используется cmix)
    :return: Длина сжатых данных в байтах
    """
    import tempfile
    import shutil

    if not shutil.which('cmix'):
        raise RuntimeError(
            "cmix не найден в PATH. Скачайте и соберите с https://github.com/byronknoll/cmix\n"
            "и добавьте бинарник в PATH."
        )

    data = _merge(INPUT_PATH_1, INPUT_PATH_2)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_in:
        tmp_in.write(data)
        tmp_in_path = Path(tmp_in.name)

    tmp_out_path = tmp_in_path.with_suffix('.bin.cmix')

    try:
        _run_subprocess(['cmix', '-c', str(tmp_in_path), str(tmp_out_path)])
        compressed_size = tmp_out_path.stat().st_size
    finally:
        if tmp_in_path.exists():
            tmp_in_path.unlink()
        if tmp_out_path.exists():
            tmp_out_path.unlink()

    return compressed_size


def _compute_nll_bits(
    model,
    tokenizer,
    tokens: "torch.Tensor",
    device: str,
    max_length: int,
) -> float:
    """
    Вычисляет количество бит для кодирования токенов через NLL.

    Использует sliding window с labels=-100 маскировкой (стандартный паттерн HuggingFace).

    :param model: Языковая модель
    :param tokenizer: Токенизатор
    :param tokens: Тензор токенов [1, seq_len]
    :param device: Устройство
    :param max_length: Максимальная длина контекста модели
    :return: Количество бит
    """
    import torch

    seq_len = tokens.shape[1]

    # Edge case: too short
    if seq_len < 2:
        return 0.0

    # Stride: half of max_length, but at least 1
    stride = max(1, max_length // 2)

    total_nll_nats = 0.0
    total_tokens = 0

    tokens = tokens.to(device)

    with torch.no_grad():
        # Sliding window approach from HuggingFace perplexity docs
        for i in range(0, seq_len, stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, seq_len)
            trg_len = end_loc - i  # Number of tokens to predict in this window

            input_ids = tokens[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            # Mask tokens we've already scored (or context tokens)
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            # outputs.loss is mean NLL in nats over non-masked tokens
            nll_nats = outputs.loss.item() * trg_len

            total_nll_nats += nll_nats
            total_tokens += trg_len

            if end_loc >= seq_len:
                break

    # Convert nats to bits: bits = nats / ln(2)
    total_bits = total_nll_nats / math.log(2)
    return total_bits


def ts_zip_compressor(INPUT_PATH_1: str | Path, INPUT_PATH_2: str | Path | None = None, lvl: int = 3) -> int:
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

    model_names = {1: 'gpt2', 2: 'gpt2-medium', 3: 'gpt2-large'}
    model_name = model_names.get(lvl, 'gpt2')

    data = _merge(INPUT_PATH_1, INPUT_PATH_2)
    text = _decode_text(data, strict=False)

    if not text:
        return 1  # Минимальный размер

    device = _get_device()
    tokenizer, model = _get_cached_model(model_name, GPT2LMHeadModel, GPT2Tokenizer, device=device)

    tokens = tokenizer.encode(text, return_tensors='pt')

    if tokens.shape[1] < 2:
        return 1  # Need at least 2 tokens to compute NLL

    max_length = _get_max_length(model)
    total_bits = _compute_nll_bits(model, tokenizer, tokens, device, max_length)

    # Возвращаем размер в байтах (ceiling)
    compressed_size = max(1, math.ceil(total_bits / 8))
    return compressed_size


def llm_compress(INPUT_PATH_1: str | Path, INPUT_PATH_2: str | Path | None = None, lvl: int = 3) -> int:
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
    text = _decode_text(data, strict=False)

    if not text:
        return 1

    device = _get_device()
    tokenizer, model = _get_cached_model(model_name, AutoModelForCausalLM, AutoTokenizer, device=device)

    tokens = tokenizer.encode(text, return_tensors='pt', truncation=False)

    if tokens.shape[1] < 2:
        return 1

    max_length = _get_max_length(model)
    total_bits = _compute_nll_bits(model, tokenizer, tokens, device, max_length)

    compressed_size = max(1, math.ceil(total_bits / 8))
    return compressed_size
