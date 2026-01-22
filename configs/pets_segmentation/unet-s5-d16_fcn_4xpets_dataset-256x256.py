# Наследуемся от базовых конфигов 
# Датасет и гиперпараметры мы подготовили на прошлых этапах
# Архитектуру используем без изменений 
_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', 
    '../_base_/datasets/pets_dataset.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_pets_baseline.py'
]

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),      # сохраняем логи локально
        dict(
            type='ClearMLVisBackend',      # дублируем всё в ClearML
            init_kwargs=dict(
                project_name='pet-segmentation',
                task_name='unet-s5-d16_fcn_4xpets_dataset-256x256',
                reuse_last_task_id=False,
                continue_last_task=False,
                output_uri=None,
                auto_connect_arg_parser=True,
                auto_connect_frameworks=True,
                auto_resource_monitoring=True,
                auto_connect_streams=True,
            )
        )     
    ]
)


# Определим размер входа 
input_suze = (256, 256)
# Если вы посмотрите на другие конфиги то можете заметить, что там часто на больших разрешениях
# ипользуется более сложный препроцессинг, когда изображение бьётся на части 
# и каждый кусочек инферится по очереди
# Здесь мы будем использовать наиболее простой метод - инференс всего изображения сразу 
data_preprocessor = dict(size=input_suze)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode="whole")
)