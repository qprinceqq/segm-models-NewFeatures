"""
Скрипт для конвертации чекпоинта проекта segm-models в .onnx модель
"""

import argparse
import torch
import numpy as np
# from torchsummary import summary
from model import get_model
from torch import nn
import torch.onnx
import onnx
from glob import glob
import os
import onnxruntime
import sys


def convert_checkpoint_to_onnx(check_path: str, output_path: str, device: str):
    checkpoint = torch.load(check_path, map_location=torch.device(device))
    model_name = checkpoint['model_name']
    encoder_name = checkpoint['encoder_name']
    encoder_weights = checkpoint['encoder_weights']
    activation = checkpoint['activation']
    in_channels = 3
    add_dirs = checkpoint['add_dirs']
    if add_dirs is not None:
        in_channels += len(add_dirs)
    model = get_model(model_name=model_name, encoder_name=encoder_name, encoder_weights=encoder_weights,
                      activation=activation, in_channels=in_channels)
    model.encoder.set_swish(memory_efficient=False)

    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.module.to(device)  # убираем nn.DataParallel т.к. с ним не считается на cpu
    model.eval()

    # for printing
    # model.to('cuda')
    # # print(model)
    # input = torch.tensor((3, 512, 512))
    # summary(model, (3, 512, 512))
    # exit()

    batch_size = 1
    x = torch.randn(batch_size, in_channels, 512, 512, requires_grad=True)
    torch_out = model(x)
    # print(f'torch out {torch_out}')

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      output_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=13,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(output_path)

    def to_numpy(tensor):
        # return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        return tensor.detach().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"{torch_out.shape}")
    print(f"{ort_outs[0].shape}")
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def main():
    parser = argparse.ArgumentParser(description='Run model in evaluation mode')
    parser.add_argument("-c", "--check-path", type=str, help="checkpoint path")
    parser.add_argument("-d", "--dir", type=str, help="path to dir with checkpoints")
    parser.add_argument("-o", "--output-path", type=str, help="path where to write output, if isn't provided")
    parser.add_argument("--device", type=str, default="cpu", help="default=cuda, or cpu")
    # parser.add_argument("-nc", "--num-channels", type=int, default=3, help="3 or 4. Num channels of an input image")
    args = parser.parse_args()
    print("Passed arguments: ", args)

    # УБРАТЬ ЕСЛИ НУЖНО ЗАПУСКАТЬ СКРИПТ ИЗ КОНСОЛИ
    # args.check_path = r'D:\Работа\Студенты\2024\Константин Облака\clouds_OnCloudN_20m\efficientnet-b0_bsize_30_hard\deeplabv3+_efficientnet-b0_best_model.pth'
    # args.output_path = r'.\onnx\clouds.onnx'
    args.dir = r'D:\Работа\Студенты\2024\Бобров Михаил\Trees_512_bsize_128_b7_hard_JaccardLoss_adamW_8gpu'
    args.output_path = r'D:\Работа\Студенты\2024\Бобров Михаил\Trees_512_bsize_128_b7_hard_JaccardLoss_adamW_8gpu'

    if args.output_path is not None:  # Создаем папку куда сохраняем результаты
        os.makedirs(args.output_path, exist_ok=True)

    if args.dir is None and args.check_path is None:
        print('Please provide --dir or --check-path', file=sys.stderr)
        exit(1)

    if args.dir is not None:
        check_path = glob(os.path.join(args.dir, '*.pth'))
        if len(check_path) == 0:
            print(f'No checkpoint found in {args.dir}', file=sys.stderr)
            exit(1)
    else:
        check_path = [args.check_path]

    ext = '.onnx'
    output_path = args.output_path
    for check in check_path:
        print(f'Processing {check}')
        if args.output_path is None:
            output_path, check_filename = os.path.split(check)
            check_filename, _ = os.path.splitext(check_filename)
            output_path = os.path.join(output_path, check_filename + ext)
        elif args.dir is not None: # output_path не пустой и мы обрабатываем папку
            _, check_filename = os.path.split(check)
            check_filename, old_ext = os.path.splitext(check_filename)
            output_path = os.path.join(args.output_path, check_filename + ext)

        convert_checkpoint_to_onnx(check, output_path, args.device )


if __name__ == '__main__':
    main()
