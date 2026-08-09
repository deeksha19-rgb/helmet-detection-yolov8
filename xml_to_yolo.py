import xml.etree.ElementTree as ET
from pathlib import Path

IMAGES_DIR = "/content/dataset/dataset/images"
ANNOTATIONS_DIR = "/content/dataset/dataset/annotations"
LABELS_OUT_DIR = "/content/dataset/dataset/labels"

def discover_classes(xml_files):
    names = set()
    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            names.add(obj.find("name").text.strip())
    classes = sorted(names)
    print(f"Discovered classes: {classes}")
    return classes

def convert_bbox(img_w, img_h, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h

def convert_one(xml_path, out_dir, classes):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        cls_id = classes.index(cls_name)
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)
        x, y, w, h = convert_bbox(img_w, img_h, xmin, ymin, xmax, ymax)
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    out_path = out_dir / (xml_path.stem + ".txt")
    out_path.write_text("\n".join(lines))

def main():
    ann_dir = Path(ANNOTATIONS_DIR)
    out_dir = Path(LABELS_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {ann_dir}")
    classes = discover_classes(xml_files)
    print(f"Converting {len(xml_files)} annotation files...")
    errors = []
    for xml_path in xml_files:
        try:
            convert_one(xml_path, out_dir, classes)
        except Exception as e:
            errors.append((xml_path.name, str(e)))
    print(f"Done. {len(xml_files) - len(errors)} converted, {len(errors)} failed.")
    (out_dir.parent / "classes.txt").write_text("\n".join(classes))
    print(f"Wrote class list to {out_dir.parent / 'classes.txt'} -> {classes}")

if __name__ == "__main__":
    main()
