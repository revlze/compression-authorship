from pathlib import Path
from nltk.corpus import stopwords
import random
import json
import re

# === Некоторые константы для регулярок ====

# управляющие/нечитаемые символы (кроме \n и \t)
RE_CONTROL = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\uFFFD]')

# несколько пробелов/табов подряд -> один пробел
RE_SPACES = re.compile(r'[ \t]+')

# 3+ пустых строк подряд -> 2 (нормализуем “лестницу” из пустых строк)
RE_MANY_EMPTY_LINES = re.compile(r'\n[ \t]*\n[ \t]*\n+')

# слова / пробелы / пунктуация
TOKEN_RE = re.compile(r'(\w+|\s+|[^\w\s])', flags=re.UNICODE)
WORD_RE  = re.compile(r'^\w+$', flags=re.UNICODE)

# === стоп-слова из NLTK (русский) ===
# ВАЖНО: один раз в окружении сделай:
#   import nltk
#   nltk.download('stopwords')
STOPWORDS_RU: set[str] = set(stopwords.words('russian'))


def clean_text(text: str) -> str:
    # 1) нормализуем переводы строк
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 2) убираем управляющие/нечитаемые символы
    text = RE_CONTROL.sub('', text)

    # 3) заменяем табы и пачки пробелов на один пробел
    text = RE_SPACES.sub(' ', text)

    # 4) нормализуем “ступеньки” из пустых строк
    text = RE_MANY_EMPTY_LINES.sub('\n\n', text)

    # 5) финальный trim
    return text.strip()


def preprocess_text(
    text: str,
    *,
    lowercase: bool,
    remove_punct: bool,
    remove_stopwords: bool,
    stopwords_set: set[str] | None = None,
) -> str:
    # базовая чистка
    text = clean_text(text)

    # регистр
    if lowercase:
        text = text.lower()

    if not remove_punct and not remove_stopwords:
        # ничего сложного не делаем
        return text

    if remove_stopwords and stopwords_set is None:
        raise ValueError("remove_stopwords=True, но stopwords_set=None")

    tokens_out: list[str] = []

    for tok in TOKEN_RE.findall(text):
        # пробелы и переводы строк всегда оставляем
        if tok.isspace():
            tokens_out.append(tok)
            continue

        # слово (буквы/цифры/_) — тут решаем про стоп-слова
        if WORD_RE.match(tok):
            if remove_stopwords and tok.lower() in stopwords_set:
                # просто выкидываем
                continue
            tokens_out.append(tok)
            continue

        # иначе — пунктуация / прочие символы
        if not remove_punct:
            tokens_out.append(tok)

    result = ''.join(tokens_out)

    # на всякий случай ещё раз подровняем пробелы и пустые строки
    result = RE_SPACES.sub(' ', result)
    result = RE_MANY_EMPTY_LINES.sub('\n\n', result)
    return result.strip()


def build(files, target_kb, min_chunk_kb=4, rng=None):
    """
    Собирает смесь кусков из произведений.
    Размер куска пропорционален длине произведения.
    Для длинных произведений размер не меньше min_chunk_kb.
    Для коротких произведений берётся текст целиком.
    """
    if rng is None:
        rng = random

    target_bytes = target_kb * 1024
    min_chunk_bytes = min_chunk_kb * 1024

    # все в байты перевел
    byte_files = {name: text.encode("utf-8") for name, text in files.items()}

    # длины произведений
    lengths = {name: len(byte_text) for name, byte_text in byte_files.items()}
    total = sum(lengths.values())

    # считаем пропорции
    props = {}
    for name, L in lengths.items():
        if L <= min_chunk_bytes:
            # короткое произведение — берём целиком
            size = L
        else:
            # доля
            prop = L / total

            size = int(prop * target_bytes)
            size = max(size, min_chunk_bytes)
            size = min(size, int(L * 0.9))

        props[name] = size

    # если сумма > target_bytes, то уменьшаем
    sum_sizes = sum(props.values())
    if sum_sizes > target_bytes:
        ratio = target_bytes / sum_sizes
        for name, size in list(props.items()):
            L = lengths[name]
            if L <= min_chunk_bytes:
                # короткие тексты не трогаем
                props[name] = L
            else:
                new_size = int(size * ratio)
                new_size = max(min_chunk_bytes, min(new_size, int(L * 0.9)))
                props[name] = new_size

    output_chunks = []
    for name, byte_text in byte_files.items():
        L = len(byte_text)
        need = props[name]

        if need >= L:
            chunk = files[name].strip()
        else:
            # выбираем случайную стартовую позицию
            start = rng.randint(0, L - need)
            sub = byte_text[start:start+need].decode("utf-8", errors="ignore")

            # перенос начала до ближайшего абзаца
            idx = sub.find("\n\n")
            if idx != -1:
                sub = sub[idx+2:]
            chunk = sub.strip()

        output_chunks.append(chunk)

    return "\n\n".join(output_chunks)


def process(
    author_name,
    train_kb,
    test_kb,
    *,
    seed=2025,
    min_chunk_kb=4,
    lowercase: bool = False,
    remove_punct: bool = False,
    remove_stopwords: bool = False,
):
    raw_fold = Path("data/raw") / author_name
    base_processed_fold = Path("data/processed") / author_name

    if not raw_fold.exists():
        raise FileNotFoundError(f"Папка с исходниками не найдена: {raw_fold}, советую проверить имя автора")

    base_processed_fold.mkdir(parents=True, exist_ok=True)

    # подготавливаем сет стоп-слов, если нужно 
    stopwords_set: set[str] | None = STOPWORDS_RU if remove_stopwords else None

    # читаем сырые тексты
    list_txt_files = list(raw_fold.glob('*.txt'))
    raw_txt_files = {p.stem: p.read_text(encoding='utf-8') for p in list_txt_files}

    # проверка кол-ва файлов — тут у тебя была ошибка (txt_files ещё нет)
    if len(raw_txt_files) < 2:
        raise ValueError(f"Нужно хотя бы 2 произведения, найдено {len(raw_txt_files)}")

    # применяем препроцессинг ко всем произведениям
    txt_files = {
        title: preprocess_text(
            text,
            lowercase=lowercase,
            remove_punct=remove_punct,
            remove_stopwords=remove_stopwords,
            stopwords_set=stopwords_set,
        )
        for title, text in raw_txt_files.items()
    }

    # имя подпапки под конкретную конфигурацию (чтобы не затирать результаты)
    cfg_parts = []
    cfg_parts.append("lower" if lowercase else "origcase")
    cfg_parts.append("nopunct" if remove_punct else "withpunct")
    cfg_parts.append("nostop" if remove_stopwords else "withstop")
    cfg_name = "_".join(cfg_parts)

    processed_fold = base_processed_fold / cfg_name
    processed_fold.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # делим произведения на train и test
    titles = list(txt_files.keys())
    rng.shuffle(titles)

    split_point = len(titles) * 2 // 3  # 2/3 произведений кидаем в train, чтоб он был разнообразнее
    train_titles = titles[:split_point]
    test_titles  = titles[split_point:]

    train_files = {t: txt_files[t] for t in train_titles}
    test_files  = {t: txt_files[t] for t in test_titles}

    # генерируем train/test
    train_text = build(train_files, train_kb, min_chunk_kb=min_chunk_kb, rng=rng)
    test_text  = build(test_files, test_kb, min_chunk_kb=min_chunk_kb, rng=rng)

    # пути до train/test
    train_path = processed_fold / 'Train.txt'
    test_path  = processed_fold / "Test.txt"

    # сохраняем
    train_path.write_text(train_text, encoding="utf-8")
    test_path.write_text(test_text, encoding="utf-8")

    # логи
    meta = {
        "author": author_name,
        "seed": seed,
        "train_kb": train_kb,
        "test_kb": test_kb,
        "min_chunk_kb": min_chunk_kb,
        "train_titles": train_titles,
        "test_titles": test_titles,
        "lowercase": lowercase,
        "remove_punct": remove_punct,
        "remove_stopwords": remove_stopwords,
        "stopwords_source": "nltk.russian",
        "config_name": cfg_name,
    }

    (processed_fold / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Готово! Сохранено в папке {processed_fold}")
    print(f"В train попали: {train_titles}")
    print(f"В test: {test_titles}")
    return train_path, test_path, meta


# ===== Запуск 8 конфигураций =====

author = "Pasternak"

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

for cfg in configs:
    process(
        author,
        train_kb=128,
        test_kb=128,
        seed=2025,
        min_chunk_kb=4,
        **cfg,
    )
