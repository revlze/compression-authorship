from pathlib import Path
import random
import json


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


def process(author_name, train_kb, test_kb, seed=2025, min_chunk_kb=4):
    raw_fold = Path("data/raw") / author_name
    processed_fold = Path("data/processed") / author_name
    
    if not raw_fold.exists():
        raise FileNotFoundError(f"Папка с исходниками не найдена: {raw_fold}, советую проверить имя автора")

    processed_fold.mkdir(parents=True, exist_ok=True)

    # собираем все произведения в словарь {title : full_text}
    list_txt_files = list(raw_fold.glob('*.txt'))
    txt_files = {p.stem: p.read_text(encoding='utf-8') for p in list_txt_files}

    # проверка кол-ва файлов
    if len(txt_files) < 2:
        raise ValueError(f"Нужно хотя бы 2 произведения, найдено {len(txt_files)}")

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
    test_text = build(test_files, test_kb, min_chunk_kb=min_chunk_kb, rng=rng)

    # пути до train/test
    train_path = processed_fold / 'Train.txt'
    test_path = processed_fold / "Test.txt"

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
    }

    (processed_fold / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Готово! Сохранено в папке {processed_fold}")
    print(f"В train попали: {train_titles}")
    print(f"В test: {test_titles}")
    return train_path, test_path, meta

process("Pasternak", train_kb=128, test_kb=128, seed=2025)