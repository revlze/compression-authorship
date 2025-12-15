# Модуль для машинного перевода (Способ 2)
# Переводим оригинальный английский текст на русский разными переводчиками
# и сравниваем с человеческими переводами через NCD

from pathlib import Path
import time

# deep-translator поддерживает несколько движков
from deep_translator import GoogleTranslator

# transformers для Helsinki-NLP модели
try:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def _chunk_text(text: str, max_chars: int = 4500) -> list[str]:
    """
    Разбивает текст на куски для API-переводчиков (лимит ~5000 символов).
    Разбиваем по предложениям, чтобы не резать слова.
    """
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 2 < max_chars:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def translate_google(text: str, delay: float = 1.0, max_retries: int = 3) -> str:
    """
    Перевод через Google Translate (deep-translator).
    """
    translator = GoogleTranslator(source='en', target='ru')
    chunks = _chunk_text(text, max_chars=4000)
    results = []

    for i, chunk in enumerate(chunks):
        for attempt in range(max_retries):
            try:
                translated = translator.translate(chunk)
                results.append(translated)
                time.sleep(delay)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 2))  # Увеличиваем задержку
                else:
                    print(f"Google error (chunk {i}): {e}")
                    results.append("")  # Пропускаем при ошибке

        if (i + 1) % 20 == 0:
            print(f"  Google: {i + 1}/{len(chunks)} chunks...")

    return " ".join(results)


def translate_helsinki(text: str, device: str = 'cuda') -> str:
    """
    Перевод через Helsinki-NLP/opus-mt-en-ru (локальная модель).
    Работает офлайн, но требует ~300MB на первую загрузку.

    :param device: 'cuda', 'cpu' или None (авто-выбор)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers не установлен: pip install transformers sentencepiece")

    # Авто-выбор устройства
    # device = 'cuda'
    print(f"Helsinki-NLP: используется {device}")

    model_name = "Helsinki-NLP/opus-mt-en-ru"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    # Разбиваем на предложения (модель лучше работает с короткими)
    sentences = text.replace('\n', ' ').split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    # Батчим по 8 предложений (можно увеличить на GPU)
    batch_size = 16 if device == "cuda" else 8
    results = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Переносим входы на то же устройство, что и модель
        inputs = {k: v.to(device) for k, v in inputs.items()}
        translated = model.generate(**inputs)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        results.extend(decoded)

    return " ".join(results)


# Словарь доступных переводчиков
TRANSLATORS = {
    'google': translate_google,
    'helsinki': translate_helsinki,
}


def translate_file(input_path: Path, output_path: Path, translator_name: str = 'google') -> Path:
    """
    Переводит файл и сохраняет результат.

    :param input_path: путь к исходному английскому тексту
    :param output_path: путь для сохранения перевода
    :param translator_name: название переводчика из TRANSLATORS
    :return: путь к переведённому файлу
    """
    if translator_name not in TRANSLATORS:
        raise ValueError(f"Неизвестный переводчик: {translator_name}. Доступны: {list(TRANSLATORS.keys())}")

    text = input_path.read_text(encoding='utf-8')
    translator_func = TRANSLATORS[translator_name]

    print(f"Перевод {input_path.name} через {translator_name}...")
    translated = translator_func(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(translated, encoding='utf-8')
    print(f"Сохранено: {output_path}")

    return output_path


def translate_twain_all(twain_path: Path, output_dir: Path) -> dict[str, Path]:
    """
    Переводит текст Твена всеми доступными переводчиками.

    :param twain_path: путь к оригинальному английскому тексту
    :param output_dir: директория для сохранения переводов
    :return: словарь {translator_name: output_path}
    """
    results = {}

    for name in TRANSLATORS:
        output_path = output_dir / f"twain_{name}_translation.txt"

        # Пропускаем, если уже переведено
        if output_path.exists():
            print(f"Уже есть: {output_path}")
            results[name] = output_path
            continue

        try:
            results[name] = translate_file(twain_path, output_path, name)
        except Exception as e:
            print(f"Ошибка {name}: {e}")

    return results
