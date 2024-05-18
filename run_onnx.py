"""
Запуск .onnx файла на снимке:
Не умеет считать батчами
Не любит киррилицу в пути к картинкам!
run_onnx.py -m path_to_onnx -i source_img_path -o output_path
run_onnx.py -m path_to_onnx -d source_dir_path -o output_path
"""

import onnxruntime
import numpy as np
import operator
from tqdm import tqdm
import os
from argparse import ArgumentParser
import math
import cv2
from glob import glob
import sys

# import requests

# TODO Добавить корректную обработку тайлов меньше 512*512 пикселей
# Растр -> Извлечение -> Обрезать растр по обхвату

DEBUG = True
url = 'https://cloud.imm.uran.ru/s/HsMkitXrdd82SS5/download/buildings_b6.onnx'

file_exts = ('*.tif', '*.png', '*.jpg')

def normalize(img):
    """
    Вычисляет среднее и дисперсию по каждому каналу изображения, и нормализует изображение
    """
    # mean_def = (0.485, 0.456, 0.406)
    # std_def = (0.229, 0.224, 0.225)
    # max_pixel_value = 255.0
    mean = []
    std = []
    for ch in range(img.shape[2]):
        m = np.array(img[:, :, ch]).mean()
        mean.append(m)
        v = img[:, :, ch] - m
        v = np.sqrt(np.mean(v ** 2))
        std.append(v)
    print(f'Calculated mean {mean} calculated std {std}')

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img

    # mean = np.array(mean, dtype=np.float32)
    # mean *= max_pixel_value
    # std = np.array(std, dtype=np.float32)
    # std *= max_pixel_value
    # denominator = np.reciprocal(std, dtype=np.float32)
    # img = img.astype(np.float32)
    # img -= mean
    # img *= denominator
    # return img


def run_network(img, session):
    # compute ONNX Runtime output prediction
    ort_inputs = {session.get_inputs()[0].name: img}
    return session.run(None, ort_inputs)[0]


# def download_model(url, model_path):
#     response = requests.get(url, stream=True)
#
#     with open(model_path, "wb") as handle:
#         for data in tqdm(response.iter_content()):
#             handle.write(data)

def generate_weight_for_pos(x: int, y: int, step: int):
    """
    step = tile_size // 2
    """
    cos_x = 0.5 + 0.5*np.cos(math.pi * (x - step) / step)
    cos_y = 0.5 + 0.5*np.cos(math.pi * (y - step) / step)
    return cos_x * cos_y


def calculate_iou(gt_mask, pred_mask, eps=1e-7):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask) + eps
    iou_score = (np.sum(intersection) + eps) / np.sum(union)
    return iou_score


def calculate_accuracy(gt_mask, pred_mask):
    matching_pixels = np.sum(gt_mask == pred_mask)
    total_pixels = gt_mask.size
    accuracy = matching_pixels / total_pixels
    return accuracy


def calc_metrics(res, gt_path: str, label=255, thres=0.5):
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        print(f"CAN'T READ gt from {gt_path}", file=sys.stderr)
        return None

    # res = cv2.imread(r'D:\Vector_data\gt\frag.tif', cv2.IMREAD_GRAYSCALE)
    # res = np.array(res)
    # res = np.where(res >= 255, 1, 0)

    res = np.where(res >= thres, 1, 0)
    gt = np.array(gt)
    gt = np.where(gt == label, 1, 0)

    iou = calculate_iou(gt, res)
    acc = calculate_accuracy(gt, res)
    print(f'IoU: {iou}', f'Acc: {acc}')
    return {'iou': float(iou), 'acc': float(acc)}


def process_image(img_path: str, num_channels: int, batch_size: int, tile_size: int, step:int, model, output_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = np.asarray(img)
    img = img[:, :, 0:num_channels]
    print(img.shape)
    img = normalize(img)

    img = np.moveaxis(img, -1, 0)
    c, w, h = img.shape

    img = img.reshape((1,) + img.shape)
    print(f'img shape is {img.shape}')

    res_shape = tuple(map(operator.add, img.shape, (0, 0, 3 * step - w % step, 3 * step - h % step)))
    print(f'Inner shape is {res_shape}')

    res = np.zeros(res_shape, dtype=np.float32)
    res[:, :, step:step + w, step:step + h] = img
    img = res

    w_new = res_shape[2]
    h_new = res_shape[3]
    res = np.zeros((1, w_new, h_new), dtype=np.float32)

    tile_weights = np.fromfunction(lambda i, j: generate_weight_for_pos(i, j, step), (tile_size, tile_size),
                                   dtype=float)

    # tile_weights[:, 0: tile_size//2] += tile_weights[:, tile_size//2: tile_size]
    # cv2.imshow("mask", tile_weights)
    # cv2.imwrite("weighs_sum.tif",tile_weights)
    # cv2.waitKey(0)

    cols = (w_new // step) - 1
    rows = (h_new // step) - 1
    k_max = cols * rows  # общее число тайлов
    batch = np.zeros((batch_size,c, tile_size, tile_size), dtype=np.float32)
    k=0
    # m - число элементов в строке, то a[i*m+j] эквивалентно a[i][j]
    with tqdm(total=k_max) as pbar:
        while k < k_max:
            cur_batch_size = min(k+batch_size,k_max) - k
            for b in range(cur_batch_size):
                cur_idx = k+b
                i = (cur_idx // rows)*step
                j = (cur_idx % rows)*step
                # print(f'i is {i} j is {j}')
                batch[b,:,:,:] = img[0,:, i:i + tile_size, j:j + tile_size]
            out = run_network(batch, model)
            for b in range(cur_batch_size):
                cur_idx = k + b
                i = (cur_idx // rows)*step
                j = (cur_idx % rows)*step
                res[0, i:i + tile_size, j:j + tile_size] += out[b,0,:,:] * tile_weights
                # res[0,:, i:i + tile_size, j:j + tile_size] = batch[b,:,:,:]
            k += cur_batch_size
            pbar.update(cur_batch_size)


    img = res.reshape((w_new, h_new))
    img = img[step:step + w, step:step + h]

    # cv2.imwrite("weighs_full.tif", img)

    max_val = np.max(img)
    min_val = np.min(img)

    result = 255.0 * (img - min_val) / (max_val - min_val)
    print(f'result shape is {result.shape}')
    # print(f'max_val val is {max_val} min_val val is {min_val}')
    cv2.imwrite(output_path, np.uint8(result))
    print(f"Result was written in {output_path}")

    return img


def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="path to model weights")
    parser.add_argument("-md", "--model-dir", type=str, help="path to dir with models")
    parser.add_argument("-i", "--image", type=str, help="image path for processing")
    parser.add_argument("-d", "--dir", type=str, help="path to dir with images")
    parser.add_argument("-o", "--out", type=str, help="path for output")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="size of batch")
    parser.add_argument("--auto-label", action='store_true', help="get label from a checkpoint's name")
    parser.add_argument("-cl", "--class-label", type=int, default=255, help="class label in gt")
    # parser.add_argument("-w", "--workers", type=int, default=6, help="default=6")
    parser.add_argument('--cpu', action='store_true', help="use cpu for training")
    parser.add_argument("--device", type=str, default="cuda", help="default=cuda, or cpu")
    args = parser.parse_args()

    if args.cpu:
        args.device = 'cpu'

    print("Passed arguments: ", str(args).replace(',', ',\n'))

    if not (args.image or args.dir):
        print(f'Please provide --image or --dir for processing')
    else:
        print(f'Path for processing {args.image if args.image is not None else args.dir}')

    # print("Downloading model weights...")
    # download_model(url, model_path)
    if args.model_dir is not None:
        models_path = glob(os.path.join(args.model_dir, '*.onnx'))
        if len(models_path) == 0:
            print(f'No checkpoint found in {args.model_dir}', file=sys.stderr)
            exit(1)
    else:
        models_path = [args.model]

    metrics_path = None
    gt_path = None
    if args.dir is not None:  # значит передали в dir и out папку, создадим out
        if args.out is None:
            # path = os.path.split(args.dir)[0]
            args.out = os.path.join(args.dir, 'out')
        exist_ok = True
        os.makedirs(args.out, exist_ok=exist_ok)
        metrics_path = args.out

        gt_path = os.path.join(args.dir, "gt")
        _b_calc_metrics = os.path.isdir(gt_path)  # проверяем, что есть папка с разметкой
        if not _b_calc_metrics:
            print(f"Can't find 'gt' {gt_path}. THE SCRIPT CAN'T CALCULATE METRICS!!!", file=sys.stderr)
    else:
        output_path, img_filename = os.path.split(args.image)
        metrics_path = output_path
        if args.out is None:  # зададим путь, куда запишем результат
            img_name, ext = os.path.splitext(img_filename)
            postfix = '_out'
            args.out = os.path.join(output_path, img_name + postfix + ext)

    metrics_path = os.path.join(metrics_path, "_metrics.csv")
    with open(metrics_path, "a") as metrics_file:
        metrics_names = 'model; file; iou; accuracy \n'  # если изменили порядок вычисления метрик в make_prediction,
        metrics_file.write(metrics_names)  # печатаем шапку .csv файла

    for path in models_path:
        if args.auto_label:  # Значит метка класса указана в имени чекпоинта
            class_label = path.rsplit('_', 1)[1]
            class_label = class_label.split('.')[0]
            args.class_label = int(class_label)
        print(f'CLASS LABELS is {args.class_label}')
        # TODO писать индекс чекпоинта или tqdm
        print(f'Run {path}')
        _, check_filename = os.path.split(path)
        check_name, _ = os.path.splitext(check_filename)

        sess_opt = onnxruntime.SessionOptions()
        sess_opt.intra_op_num_threads = 8
        sess_opt.inter_op_num_threads = 8

        model = onnxruntime.InferenceSession(path, sess_opt, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        num_channels = model.get_inputs()[0].shape[1]
        tile_size = model.get_inputs()[0].shape[2]
        step = tile_size // 2

        img_path = args.image
        image_list = []

        if args.dir is not None:  # значит передали в dir и out папку, создадим out
            for ext in file_exts:
                image_list.extend(glob(os.path.join(args.dir, ext)))

            for image in image_list:
                print(f'Processing {image}')
                _, img_filename = os.path.split(image)
                img_name, ext = os.path.splitext(img_filename)
                out_path = os.path.join(args.out, img_name+'_'+check_name+ext)
                result = process_image(image, num_channels, args.batch_size, tile_size, step, model, out_path)
                if _b_calc_metrics:
                    res = calc_metrics(result, os.path.join(gt_path,img_filename), thres=0.5, label=args.class_label)

                    with open(metrics_path, "a") as metrics_file:
                        metrics_file.write(f"{check_name}; {img_filename}; {res['iou']:.4f}; {res['acc']:.4f}\n".replace('.', ','))

        else:  # значит передали одну картинку
            result = process_image(args.image, num_channels, args.batch_size, tile_size, step, model, args.out)

            gt_path = os.path.join(output_path, "gt")
            _b_calc_metrics = os.path.isdir(gt_path)  # проверяем, что есть папка с разметкой
            if not _b_calc_metrics:
                print(f"Can't find 'gt' {gt_path}. THE SCRIPT CAN'T CALCULATE METRICS!!!", file=sys.stderr)
            else:
                calc_metrics(result, os.path.join(gt_path, img_filename), thres=0.5, label=args.class_label)


    if not DEBUG:
        # скопируем данные по extent и проекции из исходного файла в целевой файл
        print(f'stage fix extent')
        import subprocess
        dirname = os.path.dirname(__file__)
        subprocess.check_call([sys.executable, os.path.join(dirname, 'gdalcopyproj.py'), img_path, output_path])


if __name__ == "__main__":
    main()
