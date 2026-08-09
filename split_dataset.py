import random, shutil
from pathlib import Path

IMAGES_DIR = Path("/content/dataset/dataset/images")
LABELS_DIR = Path("/content/dataset/dataset/labels")
OUTPUT_DIR = Path("/content/dataset_split")
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.20, 0.10
SEED = 42

def main():
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6
    image_files = sorted(IMAGES_DIR.glob("*.png")) + sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.jpeg"))
    paired = [p for p in image_files if (LABELS_DIR / (p.stem + ".txt")).exists()]
    if not paired:
        raise FileNotFoundError("No matched image/label pairs found. Run xml_to_yolo.py first.")
    random.seed(SEED)
    random.shuffle(paired)
    n = len(paired)
    n_train, n_val = int(n * TRAIN_RATIO), int(n * VAL_RATIO)
    splits = {"train": paired[:n_train], "val": paired[n_train:n_train+n_val], "test": paired[n_train+n_val:]}
    for split_name, files in splits.items():
        img_out = OUTPUT_DIR / "images" / split_name
        lbl_out = OUTPUT_DIR / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path in files:
            label_path = LABELS_DIR / (img_path.stem + ".txt")
            shutil.copy(img_path, img_out / img_path.name)
            shutil.copy(label_path, lbl_out / label_path.name)
        print(f"{split_name}: {len(files)} images")
    classes_file = LABELS_DIR.parent / "classes.txt"
    class_names = classes_file.read_text().splitlines() if classes_file.exists() else []
    yaml_content = (f"path: {OUTPUT_DIR.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
                     f"nc: {len(class_names)}\nnames: {class_names}\n")
    (OUTPUT_DIR / "data.yaml").write_text(yaml_content)
    print(f"Wrote {OUTPUT_DIR / 'data.yaml'}")

if __name__ == "__main__":
    main()
