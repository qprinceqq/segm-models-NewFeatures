import segmentation_models_pytorch.utils as smp_utils
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_toolbelt.utils import count_parameters
import os
from datetime import datetime
from model import get_model
# from torchsummary import summary
from copy import deepcopy


class SegmentationTrainer:
    def __init__(
            self,
            train_set,
            valid_set,
            valid_set_list,
            # Если не False значит в valid_set_list на 0 позиции val set из набора
            use_only_add_val=False,  # Если True значит в этом списке только доп. валидационные наборы
            # и по val набору ничего не считаем
            add_val_freq=1,
            exp_name='',
            log_dir='../logs/',
            model_name='unet',
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            loss_name="DiceLoss",
            optimizer_name='AdamW',
            scheduler_name='ReduceLROnPlateau',
            activation='sigmoid',
            device='cuda',
            epochs_count=50,
            learning_rate=0.001,
            train_batch_size=2,
            valid_batch_size=2,
            train_workers_count=1,
            valid_workers_count=1,
    ):
        self.encoder_name = encoder_name
        self.model_name = model_name
        self.encoder_weights = encoder_weights
        self.optimizer_name = optimizer_name
        self.activation = activation
        self.device = device
        self.epochs_count = epochs_count
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.train_set = train_set
        self.use_only_add_val = use_only_add_val
        self.add_val_freq = add_val_freq

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        if not exp_name:
            self.exp_name = model_name + ' ' + encoder_name + \
                            ' ' + str(datetime.date(datetime.now()))
        print(f'Num classes to train: {train_set.num_classes}')
        self._model = get_model(model_name=model_name,
                                encoder_name=encoder_name,
                                encoder_weights=encoder_weights,
                                activation=activation,
                                classes=train_set.num_classes,
                                in_channels=train_set.in_channels
                                )

        print("  Parameters     :", count_parameters(self._model))
        if self.device == 'cpu':
            self._model.to(self.device)  # с nn.DataParallel не считается на cpu!
        else:
            self._model = nn.DataParallel(self._model)

        # summary(self._model, (3, 128, 128), device='cpu')

        self._train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_workers_count,
            pin_memory=self.device
        )
        # print(f'Shape of batch {train_set[0]}')
        num_elements = 0
        self.valid_loader = {  # основной val loader
            'set': DataLoader(
                valid_set,
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=valid_workers_count,
                pin_memory=self.device
            ),
            'name': valid_set.name,
            'weight': len(valid_set)
        }
        num_elements += len(valid_set)

        self.valid_loader_list = []
        for val_set in valid_set_list:
            self.valid_loader_list.append({
                'set': DataLoader(
                    val_set,
                    batch_size=valid_batch_size,
                    shuffle=False,
                    num_workers=valid_workers_count,
                    pin_memory=self.device
                ),
                'name': val_set.name,
                'weight': len(val_set)
            })
            num_elements += len(val_set)
        for loader in self.valid_loader_list:
            loader['weight'] /= num_elements

        self.valid_loader['weight'] /= num_elements

        if loss_name == "JaccardLoss":
            self._loss = smp_utils.losses.JaccardLoss()
            self.loss_name = 'jaccard_loss'
        elif loss_name == "DiceLoss":
            self._loss = smp_utils.losses.DiceLoss()
            self.loss_name = 'dice_loss'
        else:
            raise RuntimeError(f"Wrong loss name {loss_name}")

        self._metrics = [
            smp_utils.metrics.IoU(threshold=0.5),
        ]

        self._optimizer = None
        self.set_optim_by_name(optimizer_name, learning_rate)

        self.b_pass_loss_to_scheduler = False
        if scheduler_name == "CosineAnnealingLR":
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self.epochs_count)
        elif scheduler_name == "ReduceLROnPlateau":
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer)
            self.b_pass_loss_to_scheduler = True
        else:
            raise RuntimeError("Please specify a correct scheduler name")

        self._train_epoch = smp_utils.train.TrainEpoch(
            self._model,
            loss=self._loss,
            metrics=self._metrics,
            optimizer=self._optimizer,
            device=self.device,
            verbose=True,
        )

        self._valid_epoch = smp_utils.train.ValidEpoch(
            self._model,
            loss=self._loss,
            metrics=self._metrics,
            device=self.device,
            verbose=True,
        )

    def set_optim_by_name(self, optimizer_name: str, learning_rate: float):
        if optimizer_name == 'Adam':
            self._optimizer = torch.optim.Adam([
                dict(params=self._model.parameters(), lr=learning_rate),
            ])
        elif optimizer_name == 'Adadelta':
            self._optimizer = torch.optim.Adadelta([
                dict(params=self._model.parameters(), lr=learning_rate),
            ])
        elif optimizer_name == 'RMSprop':
            self._optimizer = torch.optim.RMSprop([
                dict(params=self._model.parameters(), lr=learning_rate),
            ])
        elif optimizer_name == 'SparseAdam':
            self._optimizer = torch.optim.SparseAdam([
                dict(params=self._model.parameters(), lr=learning_rate),
            ])
        elif optimizer_name == 'AdamW':
            self._optimizer = torch.optim.AdamW([
                dict(params=self._model.parameters(), lr=learning_rate),
            ])
        elif optimizer_name == 'SGD':
            self._optimizer = torch.optim.SGD([
                dict(params=self._model.parameters(), lr=learning_rate),
            ])
        elif optimizer_name == 'LBFGS':
            self._optimizer = torch.optim.LBFGS([
                dict(params=self._model.parameters(), lr=learning_rate),
            ])
        elif optimizer_name == 'ASGD':
            self._optimizer = torch.optim.ASGD([
                dict(params=self._model.parameters(), lr=learning_rate),
            ])
        elif optimizer_name == 'Adamax':
            self._optimizer = torch.optim.Adamax([
                dict(params=self._model.parameters(), lr=learning_rate),
            ])
        elif optimizer_name == 'Adagrad':
            self._optimizer = torch.optim.Adagrad([
                dict(params=self._model.parameters(), lr=learning_rate),
            ])
        else:
            raise RuntimeError(f"Can't recognise optimizer name {optimizer_name}")
        print(f"OPTIMIZER IS SET TO {optimizer_name}")

    def set_parameters_from_checkpoint(self, checkpoint_name):
        print(f'Load checkpoint from last checkpoint - {checkpoint_name}')
        checkpoint = torch.load(checkpoint_name)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self._loss = checkpoint['loss']

        epoch = checkpoint['epoch'] + 1
        max_score = checkpoint['max_score']

        return epoch, max_score

    def save_model(self, epoch, max_score, checkpoint_name):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_name': self.optimizer_name,
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict(),
            'loss': self._loss,
            'model_name': self.model_name,
            'encoder_name': self.encoder_name,
            'encoder_weights': self.encoder_weights,
            'activation': self.activation,
            'max_score': max_score,
            'mean_var': self.train_set.mean_var,
            'class_list': self.train_set.class_list,
            'add_dirs': self.train_set.add_dirs,
            'device': self.device,
        }, checkpoint_name)
        print(f'Last model checkpoint saved at {checkpoint_name}')

    def start_training(self):
        print(f'Запуск обучения модели с энкодером {self.encoder_name}')
        logs_path = os.path.join(self.log_dir, self.exp_name)
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)

        best_checkpoint_name = f'{logs_path}/{self.exp_name}.pth'

        # Сделал без расширения файла, чтобы не путался скрипт валидации
        last_checkpoint_name = f'{logs_path}/last_checkpoint'

        writer = SummaryWriter(logs_path)

        if self.valid_loader_list is not None:
            layout = {
                "Validation": {
                    "Loss": ["Multiline", [loader['name']
                                           + ' loss' for loader in self.valid_loader_list + [self.valid_loader]]],
                    "IOU": ["Multiline", [loader['name']
                                          + ' iou' for loader in self.valid_loader_list + [self.valid_loader]]],
                },
            }
            writer.add_custom_scalars(layout)

        max_score = 0

        if os.path.exists(last_checkpoint_name):
            start_epoch, max_score = self.set_parameters_from_checkpoint(
                last_checkpoint_name)

        elif os.path.exists(best_checkpoint_name):
            start_epoch, max_score = self.set_parameters_from_checkpoint(
                best_checkpoint_name)

        else:
            start_epoch = 1

        print(f"Cur epoch is {start_epoch}")

        if start_epoch > self.epochs_count:
            raise RuntimeError('Restored epoch is out of training bounds')

        for i in range(start_epoch, self.epochs_count + 1):
            print(f'Epoch: {i}')
            print(f'Epoch start LR: {self._optimizer.param_groups[0]["lr"]}')

            mean_val_iou = 0

            train_logs = self._train_epoch.run(self._train_loader)
            writer.add_scalar('Accuracy/train', train_logs['iou_score'], i)
            writer.add_scalar('Loss/train', train_logs[self.loss_name], i)
            writer.add_scalar('Learning rate', self._optimizer.param_groups[0]["lr"], i)

            # Валидация
            if not self.use_only_add_val:  # если в списке есть основной val набор считаем iou по нему
                valid_logs = self._valid_epoch.run(self.valid_loader['set'])
                writer.add_scalar(
                    self.valid_loader['name'] + ' iou', valid_logs['iou_score'], i)
                writer.add_scalar(
                    self.valid_loader['name'] + ' loss', valid_logs[self.loss_name], i)
                val_iou = valid_logs['iou_score']
                if self.valid_loader_list is not None:
                    mean_val_iou += valid_logs['iou_score'] * \
                                    self.valid_loader['weight']

            # Считаем mean_val_iou по нескольким наборам
            if self.valid_loader_list is not None:
                validate_now = (i % self.add_val_freq) == 0
                if validate_now:
                    for loader in self.valid_loader_list:
                        valid_logs = self._valid_epoch.run(loader['set'])
                        writer.add_scalar(
                            loader['name'] + ' iou', valid_logs['iou_score'], i)
                        writer.add_scalar(
                            loader['name'] + ' loss', valid_logs[self.loss_name], i)
                        mean_val_iou += valid_logs['iou_score'] * loader['weight']

            # считаем либо только по основному набору либо только по доп. наборам
            iou_value = mean_val_iou if self.use_only_add_val else val_iou

            with open(os.path.join(logs_path, "_iou_per_epoch_val.csv"), "a") as evalfile:
                # сохраняем в файл для русского экселя
                evalfile.write(f"{i}; {iou_value:.4f}\n".replace('.', ','))

            if self.b_pass_loss_to_scheduler:
                self._scheduler.step(valid_logs[self.loss_name])
            else:
                self._scheduler.step()
            print(f"Epoch end LR: {self._optimizer.param_groups[0]['lr']}")

            if max_score < iou_value:
                max_score = iou_value
                # checkpoint_name + '_iou_{:.2f}_epoch_{}.pth'.format(self._max_score, i))
                self.save_model(i, max_score, best_checkpoint_name)

            self.save_model(i, max_score, last_checkpoint_name)

        writer.close()
