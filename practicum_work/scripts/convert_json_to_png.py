import json
import cv2
import numpy as np
from pathlib import Path

def json_to_semantic_mask(data_root: str, class_mapping: dict):
    """
    Читает json/{split}/*.json и сохраняет как labels_fixed/{split}/*.png
    """
    data_root = Path(data_root)
    json_dir = data_root / "json"
    output_mask_dir = data_root / "labels_fixed"
    output_mask_dir.mkdir(exist_ok=True)

    class_to_id = {v: k for k, v in class_mapping.items()}
    splits = ["train", "val", "test"]

    total_saved = 0

    for split in splits:
        json_split_dir = json_dir / split
        output_split_dir = output_mask_dir / split
        output_split_dir.mkdir(exist_ok=True)

        if not json_split_dir.exists():
            print(f"⚠️ JSON-папка не найдена: {json_split_dir}")
            continue

        print(f"\n🔄 Сохранение масок: {split.upper()}")

        saved = 0
        for json_path in json_split_dir.glob("*.json"):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                h, w = data["imageHeight"], data["imageWidth"]
                mask = np.zeros((h, w), dtype=np.uint8)

                for shape in data["shapes"]:
                    label = shape["label"]
                    cls_id = class_to_id.get(label, 0)
                    if cls_id == 0:
                        continue
                    points = np.array(shape["points"], dtype=np.int32)
                    cv2.fillPoly(mask, [points], int(cls_id))

                output_path = output_split_dir / f"{json_path.stem}.png"
                cv2.imwrite(str(output_path), mask)
                saved += 1

            except Exception as e:
                print(f"❌ Ошибка обработки {json_path.name}: {e}")

        print(f"✅ {split}: сохранено {saved} масок")
        total_saved += saved

    print(f"\n🎉 Готово: обновлено {total_saved} масок в {output_mask_dir}/")


if __name__ == "__main__":
    json_to_semantic_mask(
        data_root="practicum_work/data/train_dataset_for_students",
        class_mapping={1: "cat", 2: "dog"}
    )