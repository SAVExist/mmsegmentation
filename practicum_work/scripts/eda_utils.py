import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


def check_image_mask_pairs(img_dir, mask_dir):
    img_files = {f for f in os.listdir(img_dir) if f.endswith('.jpg')}
    mask_files = {f for f in os.listdir(mask_dir) if f.endswith('.png')}

    img_stems = {os.path.splitext(f)[0] for f in img_files}
    mask_stems = {os.path.splitext(f)[0] for f in mask_files}

    missing_masks = img_stems - mask_stems
    extra_masks = mask_stems - img_stems

    if missing_masks:
        print("❌ Нет масок для изображений:")
        for stem in missing_masks:
            print(f"   {stem}.jpg → нет {stem}.png")

    if extra_masks:
        print("❌ Нет изображений для масок:")
        for stem in extra_masks:
            print(f"   {stem}.png → нет {stem}.jpg")

    if not missing_masks and not extra_masks:
        print("✅ Все файлы имеют пары.")

    return missing_masks, extra_masks


def check_image_mask_size_consistency(img_dir: str, label_dir: str, expected_size=(256, 256)) -> list:
    """
    Проверяет, что изображения и маски имеют правильный размер.
    """
    errors = []
    for img_path in Path(img_dir).glob("*"):
        if img_path.suffix.lower() not in ['.jpg', '.png']:
            continue
        label_path = Path(label_dir) / f"{img_path.stem}.png"
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            errors.append(f"Не удалось прочитать изображение: {img_path.name}")
            continue
        if mask is None:
            errors.append(f"Не удалось прочитать маску: {label_path.name}")
            continue

        if img.shape[:2] != expected_size:
            errors.append(f"Размер изображения {img_path.name} не соответствует {expected_size}: {img.shape[:2]}")
        if mask.shape != expected_size:
            errors.append(f"Размер маски {label_path.name} не соответствует {expected_size}: {mask.shape}")

    if not errors:
        print("✅ Все изображения и маски имеют правильный размер.")

    return errors


def check_mask_valid_classes(label_dir: str, valid_classes=(0, 1, 2)) -> list:
    """
    Проверяет, что в масках только разрешённые классы.
    """
    errors = []
    valid_set = set(valid_classes)
    for mask_path in Path(label_dir).glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            errors.append(f"Не удалось прочитать маску: {mask_path.name}")
            continue
        unique = np.unique(mask)
        invalid = [v for v in unique if v not in valid_set]
        if invalid:
            errors.append(f"Недопустимые значения в маске {mask_path.name}: {invalid}")
    
    if not errors:
        print("✅ Все маски имеют допустимые классы.")
    
    return errors


def check_class_overlap(label_dir: str) -> list:
    """
    Проверяет, нет ли пикселей, где одновременно присутствуют кот и собака
    (например, из-за ошибки разметки).
    """
    errors = []
    for mask_path in Path(label_dir).glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        cat_pixels = (mask == 1)
        dog_pixels = (mask == 2)
        if np.any(cat_pixels & dog_pixels):
            errors.append(f"Пересечение классов (кот и собака): {mask_path.name}")
    
    if not errors:
        print("✅ Пересечение классов не обнаружено.")
    
    return errors

def process_warnings(warnings):
    images, masks = [], []
    for warning in warnings:
        mask_path = Path(warning["file"])
        image_path = mask_path.parent.parent.parent / "img" / mask_path.relative_to(mask_path.parent.parent).with_suffix(".jpg")
        images.append(image_path)
        masks.append(mask_path)
        print(warning["warn"])
    return images, masks


def check_empty_or_full_masks(label_dir: str, min_pixels=10) -> list:
    """
    Проверяет, что маски не пустые (нет объектов) или полностью заполненные.
    """
    warnings = []
    for mask_path in Path(label_dir).glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        total_pixels = mask.size
        obj_pixels = np.sum((mask == 1) | (mask == 2))
        if obj_pixels == 0:
            warnings.append(f"Маска пустая (нет объектов): {mask_path.name}")
        elif obj_pixels == total_pixels:
            warnings.append(f"Маска заполнена полностью (возможно ошибка): {mask_path.name}")
        elif obj_pixels < min_pixels:
            warnings.append(f"Маска содержит слишком мало объектных пикселей ({obj_pixels}): {mask_path.name}")
    
    if not warnings:
        print("✅ Нет масок без объектов и масок полностью заполненных объектами.")
    
    return warnings


def check_too_many_components(label_dir: str, max_components=5) -> list:
    """
    Проверяет, не разбит ли объект (кот/собака) на слишком много фрагментов.
    """
    warnings = []
    for mask_path in Path(label_dir).glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        obj_mask = (mask == 1) | (mask == 2)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(obj_mask.astype(np.uint8), connectivity=8)
        # минус фон
        num_components = num_labels - 1
        if num_components > max_components:
            warnings.append({"file": str(mask_path), "warn": f"Слишком много фрагментов объекта ({num_components}): {mask_path.name}"})
    
    if not warnings:
        print("✅ Подозрительно фрагментированных объектов не обнаружено.")
    
    return warnings


def check_isolated_noise_pixels(label_dir: str, min_region_size=10) -> list:
    """
    Проверяет наличие мелких изолированных регионов (возможно, шум разметки).
    Использует connected components.
    """
    warnings = []
    for mask_path in Path(label_dir).glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Объединяем кота и собаку как объекты
        obj_mask = (mask == 1) | (mask == 2)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(obj_mask.astype(np.uint8), connectivity=8)

        small_regions = 0
        for i in range(1, num_labels):  # Пропускаем фон (метка 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_region_size:
                small_regions += 1

        if small_regions > 0:
            warnings.append({"file": str(mask_path), "warn": f"Маска {mask_path.name} содержит {small_regions} мелких фрагментов (возможно, шум)"})

    if not warnings:
        print("✅ Масок с шумом не обнаружено.")
    
    return warnings


def check_extreme_aspect_ratio(label_dir: str, max_aspect_ratio=8.0) -> list:
    """
    Проверяет, не является ли bounding box объекта слишком вытянутым.
    """
    warnings = []
    for mask_path in Path(label_dir).glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        obj_mask = (mask == 1) | (mask == 2)
        contours, _ = cv2.findContours(obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if min(w, h) == 0:
                continue
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > max_aspect_ratio:
                warnings.append({"file": str(mask_path), "warn": f"Высокое соотношение сторон ({aspect_ratio:.2f}): {mask_path.name}"})
                break
    
    if not warnings:
        print("✅ Очень вытянутых объектов не обнаружено.")
    
    return warnings


def check_holes_in_objects(label_dir: str, max_hole_area_ratio=0.3) -> list:
    """
    Проверяет, нет ли больших дыр внутри объекта.
    """
    warnings = []
    for mask_path in Path(label_dir).glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        obj_mask = (mask == 1) | (mask == 2)
        # Найти внешний контур
        contours, _ = cv2.findContours(obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        # Ограничивающий многоугольник
        area = cv2.contourArea(contours[0])
        if area == 0:
            continue
        # Площадь самого объекта (заполненные пиксели)
        filled_area = np.sum(obj_mask)
        hole_area_ratio = (area - filled_area) / area if area > 0 else 0
        if hole_area_ratio > max_hole_area_ratio:
            warnings.append({"file": str(mask_path), "warn": f"Много дыр в объекте (отношение: {hole_area_ratio:.2f}): {mask_path.name}"})
    
    if not warnings:
        print("✅ Больших дыр внутри объектов не обнаружено.")
    
    return warnings