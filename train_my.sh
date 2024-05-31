#!/bin/sh

source /opt/intelpython3/bin/activate
source segm_models/bin/activate

# можно поперебирать batch_size от большего к меньшему
#for (( i=50; i > 40; i=i-1 ))
#do
CPU_TRAIN=true
#BATCH=$i

DSET_NAME="Massachusetts_prepared"
MODEL="deeplabv3+"
# на какой целевой класс сейчас обучаем
CLASS="Roads"
CLASS_LIST="192"
EPOH=10
ENCODER="efficientnet-b7"
BATCH=60
IMAGE_SIZE="256"
AUGM="medium"
LEARNING_RATE=0.01
OPTIMIZER="Adam"
SCHEDULE="CosineAnnealingLR"
LOSS_FUNC="DiceLoss"

# лог обучения сохранится в файл ./logs/$ENCODER"_batch_"$BATCH"_%j" где j - номер задачи на кластере
# имя эксперимента задается ниже в этом файле exp-name=$ENCODER'_bsize_'$BATCH'_'$AUGM
# имя эксперимента используется для создания папки, где лежит сохраненная модель
# если запустить эксперимент с тем же именем, то обучение продолжится из чекопоинта
# обученная модель сохранится по следующему пути: './logs/'$CLASS'_'$DSET_NAME/exp-name

# Не трогать, переменные для переключения между cpu и gpu
SBATCH_CPU=""
PYTHON_CPU=""
if [ $CPU_TRAIN == false ]
then
  SBATCH_CPU="-p v100 --gres=gpu:v100:1 --nodelist=tesla-v100"
else
  PYTHON_CPU="--cpu"
fi

# ПРИМЕЧАНИЕ: после слова wrap вставлять все строки в одинарных кавычках
sbatch -n1 \
--cpus-per-task=18 \
--mem=45000 \
$SBATCH_CPU \
-t 20:00:00 \
--job-name=segm \
--output=./logs/$DSET_NAME"_"$ENCODER"_batch_"$BATCH"_"$IMAGE_SIZE"_"$AUGM"_lr_"$LEARNING_RATE"_"$OPTIMIZER"_"$SCHEDULE"_"$LOSS_FUNC"_%j" \
\
--wrap="python3.9 train.py \
--dataset='/misc/home6/m_imm_freedata/Segmentation/Roads/Massachusetts_use_it' \
--batch=$BATCH \
--model=$MODEL \
--encoder=$ENCODER \
--augmentation=$AUGM \
--exp-name=$DSET_NAME'_'$ENCODER'_batch_'$BATCH'_'$IMAGE_SIZE'_'$AUGM'_lr_'$LEARNING_RATE'_'$OPTIMIZER'_'$SCHEDULE'_'$LOSS_FUNC \
-log './logs/'$CLASS'_'$DSET_NAME \
--workers=18 \
--epochs=$EPOH \
--class-list=$CLASS_LIST \
--image-size=$IMAGE_SIZE \
-lr=$LEARNING_RATE \
--optim=$OPTIMIZER \
-sc=$SCHEDULE \
--loss=$LOSS_FUNC \
$PYTHON_CPU \
"
#done
# Если все узлы заняты и хочется запустить задачу на пол часа на отладочном узле на CPU,
#  вставить две следующие строки после слова sbatch и указать в начале этого скрипта работу на CPU
# -p debug \
# -t 00:30:00

# --workers 1