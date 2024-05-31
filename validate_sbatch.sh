#!/bin/sh

# Запуск из консоли:
# source /opt/intelpython3/bin/activate
# source segm_models/bin/activate
# CHECK_DIR="/misc/home1/u0304/segm-models/logs/water_deepglobe/efficientnet-b0_bsize_8_hard"
# DATASET_PATH="/misc/home6/m_imm_freedata/Segmentation/RG3/val"
# На CPU:
# sbatch --mem=32000 -t 10:00:00  --job-name=segm-val --output="./logs/validate_%j"  --wrap="python3.9 validate.py -cd $CHECK_DIR -d $DATASET_PATH --cpu"

# На GPU
# sbatch -p v100 --gres=gpu:v100:1 --mem=32000 -t 10:00:00  --job-name=segm-val --output="./logs/validate_%j"  --wrap="python3.9 validate.py -cd $CHECK_DIR -d $DATASET_PATH"


source /opt/intelpython3/bin/activate
source segm_models/bin/activate

CPU_TRAIN=true
CHECK_DIR="/misc/home1/u0304/segm-models/logs/water_deepglobe/efficientnet-b0_bsize_8_hard"
DATASET_PATH="/misc/home6/m_imm_freedata/Segmentation/RG3/val"
# Наборы лежат тут /misc/home6/m_imm_freedata/Segmentation

# Не трогать, переменные для переключения между cpu и gpu
SBATCH_CPU=""
PYTHON_CPU=""
if [ $CPU_TRAIN == false ]
then
  SBATCH_CPU="-p v100 --gres=gpu:v100:1"
else
  PYTHON_CPU="--cpu"
fi

# ПРИМЕЧАНИЕ: после слова wrap вставлять все строки в одинарных кавычках
sbatch \
--cpus-per-task=8 \
--mem=45000 \
$SBATCH_CPU \
-t 10:00:00 \
--job-name=segm-val \
--output=./logs/"validate_%j" \
\
--wrap="python3.9 validate.py \
-cd $CHECK_DIR \
-d $DATASET_PATH \
$PYTHON_CPU \
"
# Если все узлы заняты и хочется запустить задачу на пол часа на отладочном узле на CPU,
#  вставить две следующие строки и указать в начале этого скрипта работу на CPU
# -p debug \
# -t 00:30:00

# --workers 1
# смотрите какое num_workers рекоменуют вам в warnings в логах обучения

