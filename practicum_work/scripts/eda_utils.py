import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def analyze_class_balance(
    mask_dir: str,
    class_labels: dict,
    split_dirs: list = None,
    figsize: tuple = (14, 7),
    save_path: str = None,
    colors: dict = None  # Возможность задать цвета вручную
):
    """
    Анализирует баланс классов.
    """
    mask_dir = Path(mask_dir)
    class_ids = sorted(class_labels.keys())
    class_names = [class_labels[cls_id] for cls_id in class_ids]

    # Цвета по умолчанию (можно переопределить)
    default_colors = ['skyblue', 'lightcoral', 'gold', 'plum', 'turquoise']
    if colors is None:
        colors = {cls_name: default_colors[i % len(default_colors)] 
                 for i, cls_name in enumerate(class_names)}

    # Статистика
    total_pixels = defaultdict(float)
    total_instances = defaultdict(int)
    image_count_with_class = defaultdict(int)
    image_count = 0

    # Поиск масок
    search_dirs = [mask_dir] if split_dirs is None else [mask_dir / d for d in split_dirs]
    mask_files = []
    for search_dir in search_dirs:
        if search_dir.exists():
            mask_files.extend(search_dir.rglob("*.png"))
        else:
            print(f"⚠️ Папка не найдена: {search_dir}")

    print(f"Найдено масок: {len(mask_files)}")

    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        image_count += 1
        for cls_id in class_ids:
            if (mask == cls_id).sum() > 0:
                total_instances[cls_id] += 1
                total_pixels[cls_id] += (mask == cls_id).sum()
                image_count_with_class[cls_id] += 1

    if image_count == 0:
        print("❌ Нет данных для анализа.")
        return

    total_pixel_sum = sum(total_pixels.values())

    data = []
    avg_instances_list = []
    global_fraction_list = []
    image_fraction_list = []
    image_count_list = []

    for cls_id in class_ids:
        name = class_labels[cls_id]
        avg_instances = total_instances[cls_id] / image_count
        global_fraction = total_pixels[cls_id] / total_pixel_sum if total_pixel_sum > 0 else 0
        image_fraction = image_count_with_class[cls_id] / image_count

        data.append({
            "Класс": name,
            "Число изображений с классом": image_count_with_class[cls_id],
            "Доля изображений с классом (%)": f"{image_fraction * 100:.1f}",
            "Среднее число инстансов на изображение": f"{avg_instances:.2f}",
            "Общая доля пикселей (%)": f"{global_fraction * 100:.1f}"
        })

        avg_instances_list.append(avg_instances)
        global_fraction_list.append(global_fraction)
        image_fraction_list.append(image_fraction)
        image_count_list.append(image_count_with_class[cls_id])

    # Таблица
    df = pd.DataFrame(data)
    print("\n📊 Детальный баланс классов:")
    print(df.to_string(index=False))

    # Графики
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Анализ баланса классов", fontsize=16, weight='bold')

    # Общие параметры
    x = np.arange(len(class_names))
    width = 0.6

    # 1. Среднее число инстансов
    ax = axes[0, 0]
    bars = ax.bar(class_names, avg_instances_list,
                  color=[colors[name] for name in class_names], width=width)
    ax.set_title("Среднее число инстансов на изображение", fontsize=12)
    ax.set_ylabel("Количество", fontsize=10)
    ax.tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, avg_instances_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{val:.2f}", ha='center', va='center', fontsize=9, color='white', weight='bold')

    # 2. Доля изображений с классом
    ax = axes[0, 1]
    bars = ax.bar(class_names, image_fraction_list,
                  color=[colors[name] for name in class_names], width=width)
    ax.set_title("Доля изображений с классом", fontsize=12)
    ax.set_ylabel("Доля", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, image_fraction_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{val*100:.1f}%", ha='center', va='center', fontsize=9, color='white', weight='bold')

    # 3. Общая доля пикселей
    ax = axes[1, 0]
    bars = ax.bar(class_names, global_fraction_list,
                  color=[colors[name] for name in class_names], width=width)
    ax.set_title("Общая доля пикселей в датасете", fontsize=12)
    ax.set_ylabel("Доля", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, global_fraction_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{val*100:.1f}%", ha='center', va='center', fontsize=9, color='white', weight='bold')

    # 4. Число изображений с классом
    ax = axes[1, 1]
    bars = ax.bar(class_names, image_count_list,
                  color=[colors[name] for name in class_names], width=width)
    ax.set_title("Число изображений с классом", fontsize=12)
    ax.set_ylabel("Количество", fontsize=10)
    ax.tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, image_count_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                str(int(val)), ha='center', va='center', fontsize=9, color='white', weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Отчёт сохранён: {save_path}")

    plt.show()

    return df


def analyze_object_area_distribution(
    mask_dir: str,
    img_dir: str,
    class_labels: dict,
    split_dirs: list = None,
    bins: int = 30,
    figsize: tuple = (14, 8),
    save_path: str = None,
    colors: dict = None
):
    """
    Анализирует распределение площадей объектов по классам.
    Площадь выражена в процентах от общей площади изображения.

    :param mask_dir: Папка с масками (с подпапками train/val/test)
    :param img_dir: Папка с изображениями (чтобы узнать размер)
    :param class_labels: Словарь {id: "name"}, например {1: "cat", 2: "dog"}
    :param split_dirs: Какие подпапки анализировать, например ["train", "val"]
    :param bins: Число бинов в гистограмме
    :param figsize: Размер графика
    :param save_path: Куда сохранить изображение
    :param colors: Цвета для классов, например {"cat": "blue", "dog": "orange"}
    """
    mask_dir = Path(mask_dir)
    img_dir = Path(img_dir)
    class_ids = sorted(class_labels.keys())
    class_names = [class_labels[cls_id] for cls_id in class_ids]

    # Цвета по умолчанию
    default_colors = ['skyblue', 'lightcoral', 'gold', 'plum']
    if colors is None:
        colors = {name: default_colors[i % len(default_colors)] for i, name in enumerate(class_names)}

    # Сбор площадей (в % от изображения)
    areas_by_class = defaultdict(list)

    # Все подкаталоги
    search_dirs = [mask_dir] if split_dirs is None else [mask_dir / d for d in split_dirs]

    mask_files = []
    for search_dir in search_dirs:
        if search_dir.exists():
            mask_files.extend(search_dir.rglob("*.png"))
        else:
            print(f"⚠️ Папка не найдена: {search_dir}")

    print(f"Найдено масок: {len(mask_files)}")

    for mask_path in mask_files:
        # Определяем соответствующее изображение
        rel_path = mask_path.relative_to(mask_dir)
        img_path = img_dir / rel_path

        # Поддержка разных расширений
        if not img_path.exists():
            img_path = img_path.with_suffix(".jpg")
            if not img_path.exists():
                img_path = img_path.with_suffix(".jpeg")
                if not img_path.exists():
                    print(f"⚠️ Изображение не найдено: {img_path}")
                    continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_area = h * w

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        for cls_id in class_ids:
            cls_mask = (mask == cls_id).astype(np.uint8)
            if cls_mask.sum() == 0:
                continue

            # Найдём отдельные объекты
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)

            for i in range(1, num_labels):  # пропускаем фон (0)
                area_px = stats[i, cv2.CC_STAT_AREA]
                if area_px <= 5:
                    continue
                area_percent = (area_px / img_area) * 100  # в процентах
                class_name = class_labels[cls_id]
                areas_by_class[class_name].append(area_percent)

    if not areas_by_class:
        print("❌ Не найдено ни одного объекта.")
        return

    # Построение гистограммы
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    for name in class_names:
        if name not in areas_by_class:
            continue
        areas = np.array(areas_by_class[name])
        sns.histplot(areas, bins=bins, alpha=0.6, label=name, color=colors[name], kde=False)

    plt.xlabel("Площадь объекта (% от изображения)", fontsize=12)
    plt.ylabel("Частота", fontsize=12)
    plt.title("Распределение площадей объектов по классам", fontsize=14, fontweight='bold')
    plt.legend(title="Класс")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 30)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Гистограмма сохранена: {save_path}")

    # Таблица статистики
    data = []
    for name in class_names:
        if name not in areas_by_class:
            continue
        areas = np.array(areas_by_class[name])
        data.append({
            "Класс": name,
            "Число объектов": len(areas),
            "Средняя площадь (%)": f"{areas.mean():.2f}",
            "Медиана площади (%)": f"{np.median(areas):.2f}",
            "Min (%)": f"{areas.min():.2f}",
            "Max (%)": f"{areas.max():.2f}",
            "25% (%)": f"{np.percentile(areas, 25):.2f}",
            "75% (%)": f"{np.percentile(areas, 75):.2f}"
        })

    df = pd.DataFrame(data)
    print("\n📊 Статистика по площадям объектов (в % от изображения):")
    print(df.to_string(index=False))

    plt.show()