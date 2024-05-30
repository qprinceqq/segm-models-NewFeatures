# PyTorch Segmentation models Trainer
[Git-оригинал](https://github.com/qprinceqq/segm-models-public)
Код для обучения моделей сегментации. Модели сетей взяты из этой библиотеки: https://github.com/qubvel/segmentation_models.pytorch

 
# Как запустить модель
```
 Запустить обучение сети можно одной строкой:
 bash train_my.sh
 (В файле train_my.sh необходимо корректно указать путь до набора данных
  и метку которым покрашена разметка целевого класса в строке CLASS_LIST="192")

 Для изменения параметров запуска можно поменять их в файле train_my.sh
 Или же запустить через терминал файл train.py и передать новые параметры

 python3 train.py [-h] [-m MODEL] [--encoder ENCODER] [--image-size IMAGE_SIZE]
                [--exp-name EXP_NAME] [--class-list CLASS_LIST]
                [--max-mask-val MAX_MASK_VAL] [-b BATCH] [-lr LR] [-e EPOCHS]
                [-a AUGMENTATION] [-w WORKERS] -d DATASET
                [--add-dirs ADD_DIRS] [-log LOG_DIR] [-val VAL_DIR]
                [--add-val-dirs ADD_VAL_DIRS] [--adv-freq ADV_FREQ]
                [--use-only-add-val] [--cpu] [--device DEVICE] [--seed SEED]

                (другие параметры можно найти в файле train.py)

```
# Проделанная работа
Над проектом работали: Дубровин Руслан РИ-220942
 
                       Хренов   Егор   РИ-220910
       
[Excel-таблица проделанной работы](https://docs.google.com/spreadsheets/d/1qvftg1H0orNYrSY0jhkmTuXq1Tz0SugP8gKB6UadevE/edit?hl=ru#gid=0)
