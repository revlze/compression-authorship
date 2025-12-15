from pathlib import Path
import random
import json
import re

try:
    from nltk.corpus import stopwords
    STOPWORDS_RU_ENG: set[str] = set(stopwords.words('russian')) | set(stopwords.words('english'))
except LookupError:
    import warnings
    warnings.warn(
        "NLTK stopwords not found. Run: import nltk; nltk.download('stopwords'). "
        "remove_stopwords=True will raise an error."
    )
    STOPWORDS_RU_ENG = set()

# === Некоторые константы для регулярок ====

# управляющие/нечитаемые символы (кроме \n и \t)
RE_CONTROL = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\uFFFD]')

# несколько пробелов/табов подряд -> один пробел
RE_SPACES = re.compile(r'[ \t]+')

# 2+ пустых строк подряд -> 1 (нормализуем “лестницу” из пустых строк)
RE_MANY_EMPTY_LINES = re.compile(r'\n[ \t]*\n+')

# слова / пробелы / пунктуация
TOKEN_RE = re.compile(r'(\w+|\s+|[^\w\s])', flags=re.UNICODE)
WORD_RE  = re.compile(r'^\w+$', flags=re.UNICODE)

# строка, состоящая только из номера главы (арабская или римская), опционально с точкой
RE_CHAPTER_HEADING = re.compile(
    r'(?m)^[ \t\u00A0]*(\d+|[ivxlcdmIVXLCDM]{1,6})[ \t\u00A0]*\.?[ \t\u00A0]*$'
)

# ссылки по типу [7], [2], также не лезем во внутрь слова, типо 'слово[7]слово' --- не матчится 'слово[7]' --- матчится
RE_LINKS = re.compile(r'\[\d+\](?!\w)')

# какой-то мусор "_____" 
RE_TRASH = re.compile(r'_+')

# двойные тире "--"
RE_TIRE = re.compile(r'--+')

# примечания всякие "* дааддаа"
RE_STAR_LINE = re.compile(r'(?m)^[ \t]*\*.*$')

# строки-заголовки полностью из ЗАГЛАВНЫХ (рус/лат), цифр и знаков
RE_UPPER_LINE = re.compile(
    r'(?m)^[ \t]*[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9 \t,.:;!?«»"\'\-–—]*$'
)


# исключения для произведений. их не берем.
# exceptions_txt = ['Собачье сердце', 'Двенадцать стульев', 'Золотой теленок', 'Мертвое озеро', 'Три страны света']
# exceptions_txt = set(map(str.lower, exceptions_txt))


def clean_text(text: str) -> str:
    # нормализуем переводы строк
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # убираем управляющие/нечитаемые символы
    text = RE_CONTROL.sub('', text)

    # убираем мусор
    text = RE_TRASH.sub('\n', text)

    # нормализуем тире
    text = RE_TIRE.sub('-', text)

    # убираем номера глав (строки вида "1", "IV.", "ix" и т.п.)
    text = RE_CHAPTER_HEADING.sub('', text)

    # убираем строки-заголовки из одних ЗАГЛАВНЫХ
    text = RE_UPPER_LINE.sub('', text)

    # убираем ссылки типа [7], [12] и т.п.
    text = RE_LINKS.sub(' ', text)

    # заменяем табы и пачки пробелов на один пробел
    text = RE_SPACES.sub(' ', text)

    # нормализуем пустые строки
    text = RE_MANY_EMPTY_LINES.sub('\n', text)

    return text.strip()


def preprocess_text(
    text: str,
    *,
    lowercase: bool,
    remove_punct: bool,
    remove_stopwords: bool,
    stopwords_set: set[str] | None = None,
) -> str:
    # регистр
    if lowercase:
        text = text.lower()

    if not remove_punct and not remove_stopwords:
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

    # # на всякий случай ещё раз подровняем пробелы и пустые строки
    result = RE_SPACES.sub(' ', result)
    result = RE_MANY_EMPTY_LINES.sub('\n', result)
    return result.strip()


def build(
    files: dict[str, str],
    target_kb: int | float,
    min_chunk_kb: int | float = 4,
    rng: random.Random | None = None,
) -> str:
    """
    Собирает смесь кусков из произведений.
    Размер куска пропорционален длине произведения.
    Для длинных произведений размер не меньше min_chunk_kb.
    Для коротких произведений берётся текст целиком.
    
    Использует реальный размер в байтах (UTF-8) для точности.
    """
    if rng is None:
        rng = random.Random()

    if not files:
        return ""

    # Используем реальный размер в байтах для точности
    target_bytes = target_kb * 1024
    min_chunk_bytes = min_chunk_kb * 1024

    # длины произведений в байтах (UTF-8)
    lengths: dict[str, int] = {name: len(text.encode('utf-8')) for name, text in files.items()}
    total = sum(lengths.values())

    if total == 0:
        return ""

    # считаем пропорции (сколько байтов брать из каждого произведения)
    props: dict[str, int] = {}
    for name, L in lengths.items():
        if L <= min_chunk_bytes:
            # короткое произведение — берём целиком
            size = L
        else:
            # доля
            prop = L / total
            size = int(prop * target_bytes)
            size = max(size, int(min_chunk_bytes))
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
                new_size = max(int(min_chunk_bytes), min(new_size, int(L * 0.9)))
                props[name] = new_size

    output_chunks: list[str] = []
    for name, text in files.items():
        text_bytes = len(text.encode('utf-8'))
        need = props[name]

        if need >= text_bytes:
            chunk = text.strip()
        else:
            # Оценочное соотношение байты/символы для этого текста
            bytes_per_char = text_bytes / len(text) if text else 1
            need_chars = int(need / bytes_per_char)
            
            L = len(text)
            need_chars = min(need_chars, L)
            
            # выбираем случайную стартовую позицию (в символах)
            max_start = max(0, L - need_chars)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            sub = text[start:start + need_chars]

            # обрезаем начало до ближайшего переноса строки (начало абзаца/предложения)
            first_newline = sub.find("\n")
            if first_newline != -1 and first_newline < len(sub) // 4:
                sub = sub[first_newline + 1:]

            # обрезаем конец до ближайшего переноса строки (конец абзаца/предложения)
            # ВАЖНО: пересчитываем позицию после первой обрезки
            last_newline = sub.rfind("\n")
            if last_newline != -1 and last_newline > len(sub) * 3 // 4:
                sub = sub[:last_newline]

            # если не нашли перенос в начале, пробуем обрезать по точке/восклицательному/вопросительному
            if first_newline == -1:
                current_len = len(sub)
                for sep in (". ", "! ", "? "):
                    idx = sub.find(sep)
                    if idx != -1 and idx < current_len // 4:
                        sub = sub[idx + 2:]
                        break

            # если не нашли перенос в конце, пробуем обрезать по точке
            # ВАЖНО: пересчитываем после возможной обрезки начала
            if last_newline == -1:
                current_len = len(sub)
                for sep in (". ", "! ", "? "):
                    idx = sub.rfind(sep)
                    if idx != -1 and idx > current_len * 3 // 4:
                        sub = sub[:idx + 1]
                        break

            chunk = sub.strip()

        if chunk:  # добавляем только непустые чанки
            output_chunks.append(chunk)

    result = "\n\n".join(output_chunks)
    # нормализуем пустые строки
    result = RE_MANY_EMPTY_LINES.sub("\n", result)
    return result.strip()


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
    stopwords_set: set[str] | None = STOPWORDS_RU_ENG if remove_stopwords else None

    # читаем сырые тексты
    list_txt_files = sorted(raw_fold.glob('*.txt'), key=lambda x: x.stem) # Сортируем по алфавиту
    raw_txt_files = {p.stem: p.read_text(encoding='utf-8') for p in list_txt_files}

    # проверка кол-ва файлов
    if len(raw_txt_files) < 2 and author_name not in ['Ilf_and_Petrov_12_chairs','Ilf_and_Petrov_golden_calf','Bulgakov_experiment', 'Chukovsky_Tom', 'Engelhardt_Tom', 'Darouzes_Tom', 'Twain_machine', 'Twain_machine_google', 'Twain_machine_yandex']:
        raise ValueError(f"Нужно хотя бы 2 произведения, найдено {len(raw_txt_files)}")


    # применяем только базовую очистку (clean_text) ко всем произведениям,
    # чтобы выбор чанков был одинаковым для всех конфигураций
    txt_files = {
        title: clean_text(text)
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

    # 2/3 произведений кидаем в train, чтоб он был разнообразнее, также в некоторых случаях берем по 1/2 из-за целочисленного деления
    split_point = len(titles) * 2 // 3

    if len(raw_txt_files) < 2 and author_name in ['Ilf_and_Petrov_12_chairs','Ilf_and_Petrov_golden_calf','Bulgakov_experiment', 'Chukovsky_Tom', 'Engelhardt_Tom', 'Darouzes_Tom', 'Twain_machine', 'Twain_machine_google', 'Twain_machine_yandex']:
        split_point = 0
    
    if author_name in ['Chukovsky_Tom', 'Engelhardt_Tom', 'Darouzes_Tom']:
        split_point = 1

    train_titles = titles[:split_point]
    test_titles  = titles[split_point:]

    train_files = {t: txt_files[t] for t in train_titles}
    test_files  = {t: txt_files[t] for t in test_titles}

    # генерируем train/test (на базе очищенных, но не препроцессированных текстов)
    train_text = build(train_files, train_kb, min_chunk_kb=min_chunk_kb, rng=rng)
    test_text  = build(test_files, test_kb, min_chunk_kb=min_chunk_kb, rng=rng)

    # применяем препроцессинг (lowercase, remove_punct, remove_stopwords) ПОСЛЕ build
    train_text = preprocess_text(
        train_text,
        lowercase=lowercase,
        remove_punct=remove_punct,
        remove_stopwords=remove_stopwords,
        stopwords_set=stopwords_set,
    )
    test_text = preprocess_text(
        test_text,
        lowercase=lowercase,
        remove_punct=remove_punct,
        remove_stopwords=remove_stopwords,
        stopwords_set=stopwords_set,
    )

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
        "stopwords_source": "nltk.russian and nltk.english",
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
