# Подготовим оптимайзер, используем дефолтные параметры 
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# OptimWrapper — это обёртка над оптимизатором. Она нужна, чтобы стандартные оптимизаторы 
# и более сложные реализации из mmsegmentation имели один интерфейс, совместимый с Runner 
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),  # AdamW + низкий LR
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))

# Определяем распорядок LR
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=100,
        by_epoch=False)
]

# Определяем специфику обучающего и тренироворочного циклов
# У mmsegmentation много вариаций, как организовать обучающий цикл.
# Мы используем наиболее привычную, когда одна эпоха — это один проход по датасету
# Есть также обучающие циклы на основе итераций, в них одна эпоха — это какое-то число итераций 
# они используется с семплером, который бесконечно зацикливает датасет 
# Если вы будете смотреть на стандратные конфиги, там обычно именно такой подход.

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
# С валидационным и тестовым попроще, берём стандартные реализации
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# Указываем хуки для дополнительных возможностей, про которые мы говорили ранее 
# Обратите внимание, что аргумент определяется как default_hook 
# Дело в том, что mmsegmentation делит хуки ещё на 2 группы default и custom
# default — это наиболее базовые хуки и на первых порах мы ограничимся именно ими 
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=10, draw=True)
)