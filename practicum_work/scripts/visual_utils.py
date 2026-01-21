import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Цвета для классов: 0=фон, 1=кот (красный), 2=собака (синий)
COLOR_MAP = np.zeros((256, 3), dtype=np.uint8)
COLOR_MAP[1] = [255, 0, 0]   # Кот — красный
COLOR_MAP[2] = [0, 0, 255]   # Собака — синий


def load_mask(path: str):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # Читаем как есть
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            return mask.squeeze(axis=2)
        elif mask.shape[2] == 3:
            # Проверим, не grayscale ли это
            if np.all(mask[..., 0] == mask[..., 1]) and np.all(mask[..., 1] == mask[..., 2]):
                return mask[..., 0]
            else:
                raise ValueError(f"Цветная маска? {path}")
    raise ValueError(f"Неожиданная форма маски: {mask.shape}")


def visualize_images(images_paths, mask_paths: list, save_path=None, cols=3):
    """
    Визуализирует изображение + маска в виде таблицы.
    Маска накладывается поверх изображения.
    Подпись — имя файла.

    :param sample_pairs: Список кортежей (путь_к_изображению, путь_к_маске)
    :param cols: Количество столбцов в сетке (по умолчанию 3)
    """
    rows = (len(images_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    # Убедимся, что axes — двумерный массив
    axes = np.array(axes)
    if len(axes.shape) == 1:
        axes = axes[None, :]

    for idx, img_path in enumerate(images_paths):
        label_path = Path(mask_paths[idx])

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = load_mask(str(label_path))

        if img is None or mask is None:
            print(f"Ошибка загрузки: {img_path.name}")
            continue

        # Наложение маски
        mask_color = COLOR_MAP[mask]
        overlay = cv2.addWeighted(img, 0.7, mask_color, 0.6, 0)

        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        ax.imshow(overlay)
        ax.set_title(f"{img_path.name}", fontsize=10, pad=5)
        ax.axis("off")

    # Скрыть пустые подграфики
    for idx in range(len(images_paths), rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def compare_masks(
    img_dir: str,
    old_mask_dir: str,
    new_mask_dir: str,
    save_path: str,
    class_labels: dict = None,
    threshold: float = 256,
    cols: int = 3
):
    """
    Сравнивает старые и новые маски. Если есть различия, визуализирует с наложением.
    Сохраняет результат в файл.

    :param img_dir: Папка с изображениями
    :param old_mask_dir: Папка со старыми масками
    :param new_mask_dir: Папка с новыми масками
    :param save_path: Путь для сохранения итогового изображения
    :param class_labels: Словарь {id: "name"}, например {1: "cat", 2: "dog"}
    :param threshold: Минимальная площадь различий (в пикселях), чтобы считать маску "изменённой"
    :param cols: Количество столбцов в таблице
    """
    if class_labels is None:
        class_labels = {1: "cat", 2: "dog"}

    img_dir = Path(img_dir)
    old_mask_dir = Path(old_mask_dir)
    new_mask_dir = Path(new_mask_dir)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Цвета: старая маска — красная, новая — зелёная
    COLOR_OLD = [255, 0, 0]    # Красная
    COLOR_NEW = [0, 255, 0]    # Зелёная

    changes = []

    # Ищем все изображения
    for img_path in img_dir.rglob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        mask_name = img_path.with_suffix(".png").name
        old_mask_path = old_mask_dir / img_path.parent.name / mask_name
        new_mask_path = new_mask_dir / img_path.parent.name / mask_name

        if not old_mask_path.exists() or not new_mask_path.exists():
            print(f"⚠️ Пропуск: маска не найдена {mask_name}")
            continue

        # Загружаем
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        old_mask = cv2.imread(str(old_mask_path), cv2.IMREAD_GRAYSCALE)
        new_mask = cv2.imread(str(new_mask_path), cv2.IMREAD_GRAYSCALE)

        if old_mask is None or new_mask is None:
            continue

        # Находим различия
        diff = (old_mask != new_mask)
        if np.sum(diff) <= threshold:
            continue  # Нет существенных изменений

        # Создаём цветные маски
        overlay = img_rgb.copy()
        overlay[diff & (new_mask > 0)] = 0.6 * np.array(COLOR_NEW) + 0.4 * overlay[diff & (new_mask > 0)]
        overlay[diff & (old_mask > 0)] = 0.6 * np.array(COLOR_OLD) + 0.4 * overlay[diff & (old_mask > 0)]

        changes.append((img_rgb, overlay, img_path.name, np.sum(diff)))

    if not changes:
        print("✅ Нет изменённых масок.")
        return

    # Рисуем таблицу
    rows = (len(changes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    axes = np.array(axes)
    if len(axes.shape) == 1:
        axes = axes[None, :]

    for idx, (img_rgb, overlay, name, diff_count) in enumerate(changes):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        ax.imshow(overlay)
        ax.set_title(f"{name}\n(Изменено {diff_count} пикселей)", fontsize=9, pad=4)
        ax.axis("off")

    # Скрыть пустые
    for idx in range(len(changes), rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    
    print(f"✅ Визуализация сохранена: {save_path}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    print(f"   Показано {len(changes)} изменённых масок.")
    plt.show()

