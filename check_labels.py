from pathlib import Path


def check():
    data_root = Path('data/ET-EDU-CROPPED-PERSONS')
    concepts_path = data_root / 'concepts.txt'
    labels_path = data_root / 'train_label.txt'

    with open(concepts_path, 'r', encoding='utf-8') as f:
        concepts = [line.strip() for line in f.readlines() if line.strip()]

    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    counts = [0] * len(concepts)
    for line in lines:
        vals = list(map(int, line.split()))
        if len(vals) != len(concepts):
            continue
        for i in range(len(concepts)):
            counts[i] += vals[i]

    print(f"Tổng số ảnh Train: {len(lines)}")
    print(f"Số lớp trong concepts.txt: {len(concepts)}")
    print("-" * 40)
    for concept, count in zip(concepts, counts):
        print(f"{concept}: {count}")
    
if __name__ == '__main__':
    check()
