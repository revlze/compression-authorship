from pathlib import Path
import random

seed = 2025

random.seed(seed)

def build(files, target, min_chunk=4):
    """
    Берем кусок из каждого произведения.
    Размер куска пропорционален размеру произведения, но не меньше min_chunk.
    """
    target_bytes = target * 1024
    min_chunk_bytes = min_chunk * 1024


    # все в байты перевел
    byte_files = {name: text.encode("utf-8") for name, text in files.items()}

    # длины произведений
    lengths = {name: len(byte_text) for name, byte_text in byte_files.items()}
    total = sum(lengths.values())

    # считаем пропорции
    props = {}
    for name, L in lengths.items():
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
        for name in props:
            props[name] = max(min_chunk_bytes, int(props[name] * ratio))

    output_chunks = []
    for name, byte_text in byte_files.items():
        L = len(byte_text)
        need = props[name]

        if need >= L:
            chunk = files[name].strip()

        else:
            # выбираем случайную стартовую позицию
            start = random.randint(0, L - need)
            sub = byte_text[start:start+need].decode("utf-8", errors="ignore")

            # перенос начала до ближайшего абзаца
            idx = sub.find("\n\n")
            if idx != -1:
                sub = sub[idx+2:]
            chunk = sub.strip()

        output_chunks.append(chunk)

    return "\n\n".join(output_chunks)


def process(author_name, train_kb, test_kb):
    raw_fold = Path(f"data/raw/{author_name}")
    processed_fold = Path(f"data/processed/{author_name}")

    if not raw_fold.exists():
        raise FileNotFoundError(f"Папка с исходниками не найдена: {raw_fold}, советую проверить имя автора")

    processed_fold.mkdir(parents=True, exist_ok=True)

    # собираем все произведения в словарь {title : full_text}
    list_txt_files = list(raw_fold.glob('*.txt'))
    txt_files = {path.stem: path.read_text(encoding='utf-8') for path in list_txt_files}

    # проверка кол-ва файлов
    if len(txt_files) < 2:
        raise ValueError(f"Нужно хотя бы 2 произведения, найдено {len(txt_files)}")

    # делим произведения на train и test
    titles = list(txt_files.keys())
    random.shuffle(titles)

    split_point = len(titles) * 2 // 3  # 2/3 произведений кидаем в train, чтоб он был разнообразнее
    
    train_titles = titles[:split_point]
    test_titles  = titles[split_point:]

    train_files = {t: txt_files[t] for t in train_titles}
    test_files  = {t: txt_files[t] for t in test_titles}

    # генерируем train/test
    train_text = build(train_files, train_kb)
    test_text  = build(test_files,  test_kb)

    # пути до train/test
    train_path = processed_fold / 'Train.txt'
    test_path = processed_fold / "Test.txt"

    # сохраняем
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_text)

    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_text)

    print(f"Готово! Сохранено в папке {processed_fold}")
    print(f"В train попали: {train_titles}")
    print(f"В test: {test_titles}")

process("Pasternak", train_kb=128, test_kb=128)
