# convert/png_to_json.py
import cv2
import json
import numpy as np
from pathlib import Path
import base64

def create_shapes_from_mask(mask: np.ndarray, class_id_to_label: dict):
    shapes = []
    for cls_id, label in class_id_to_label.items():
        class_mask = (mask == cls_id).astype(np.uint8)
        if class_mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 5:
                continue
            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            points = approx.reshape(-1, 2).tolist()
            if len(points) < 3:
                continue
            shapes.append({
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })
    return shapes

def img_to_base64(img_path: Path) -> str:
    """Кодирует изображение в base64 (как делает LabelMe)"""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def png_to_json_autoload(data_root: str, class_map: dict = None):
    if class_map is None:
        class_map = {1: "cat", 2: "dog"}

    data_root = Path(data_root)
    img_dir = data_root / "img"
    mask_dir = data_root / "labels"
    json_dir = data_root / "json"
    splits = ["train", "val", "test"]

    total_converted = 0

    for split in splits:
        img_split_dir = img_dir / split
        mask_split_dir = mask_dir / split
        json_split_dir = json_dir / split
        json_split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🔄 Обработка: {split.upper()}")

        if not img_split_dir.exists():
            print(f"⚠️ Папка не найдена: {img_split_dir}")
            continue
        if not mask_split_dir.exists():
            print(f"⚠️ Папка не найдена: {mask_split_dir}")
            continue

        image_extensions = {".jpg", ".jpeg", ".png"}
        converted = 0

        for img_path in img_split_dir.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue

            mask_path = mask_split_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                print(f"  ❌ Маска не найдена: {mask_path.name}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ❌ Ошибка загрузки изображения: {img_path.name}")
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"  ❌ Ошибка загрузки маски: {mask_path.name}")
                continue

            shapes = create_shapes_from_mask(mask, class_map)

            # 🔧 Кодируем изображение в base64
            image_data = img_to_base64(img_path)

            h, w = img.shape[:2]
            json_data = {
                "version": "5.10.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": img_path.name,
                "imageData": image_data,  # ✅ Теперь не None!
                "imageHeight": h,
                "imageWidth": w,
            }

            json_path = json_split_dir / f"{img_path.stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            converted += 1

        print(f"✅ {split}: обработано {converted} файлов")
        total_converted += converted

    print(f"\n🎉 Готово: создано {total_converted} JSON-файлов в {json_dir}/")


if __name__ == "__main__":
    png_to_json_autoload(
        data_root="practicum_work/data/train_dataset_for_students",
        class_map={1: "cat", 2: "dog"}
    )
