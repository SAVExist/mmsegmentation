dataset_type = 'PetsDataset'
data_root='practicum_work/data/train_dataset_for_students'

# ==== Определяем обучающий пайплайн данных ======
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# Определяем конфиг датасета 
# По сути вместо класса мы создаём словарик с аргументами, которые мы бы подавали в конструктор
# И добавляем ещё один аргумент type, в котором указываем сам класс
train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/train',
        seg_map_path='labels_fixed/train'),
    pipeline=train_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
# Инициализируем даталоадер, в mmsegmentation много семплеров
# Мы используем стандартный  
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset
)


# ==== Определяем валидационный пайплайн данных ======
# Здесь всё аналогично обучающему
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/val',
        seg_map_path='labels_fixed/val'),
    pipeline=val_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset
)

# ==== Определяем тестовый пайплайн данных ======
test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/test',
        seg_map_path='labels_fixed/test'),
    pipeline=val_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset
)

# Здесь же в пайплайне данных создаются объекты для подсчёта метрик
# IoUMetric это общий класс для всех метрик, которые работают на уровне регионов 
# конкретные метрики указываются в виде аргумента iou_metrics
# в этом случае мы будет считать только mDice
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator