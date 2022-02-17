from argparse import ArgumentParser
import json
import os
from os.path import join as pjoin
from typing import List, Any

import cv2
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
import torch
from torch.utils.data import DataLoader
from torch_radon import RadonFanbeam
from torch_radon.radon import ParallelBeam
from torch_radon.volumes import Volume2D
from torchvision import transforms

from losses.rmse_loss import RMSELoss
from models.unet import UNet as Net
from utilities.datasets import SinoDataset
from utilities.transforms import ToTensor, ZNorm, filter_sinogram
from utilities.fan_geometry import default_fan_geometry

seed_everything(1701)


class SinoBilinUpsampling(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.parfan == 'fan':
            geom = default_fan_geometry()
            self.radon = RadonFanbeam(
                256, np.deg2rad(np.arange(360)),
                source_distance=geom.source_distance,
                det_distance=geom.det_distance,
                det_count=geom.det_count,
                det_spacing=geom.det_spacing,
            )
        elif self.hparams.parfan == 'parallel':
            self.radon = ParallelBeam(
                363, np.deg2rad(np.arange(180)),
                volume=Volume2D(256)
            )
        else:
            raise NotImplementedError()
        self.net = Net()
        self.loss = torch.nn.L1Loss()
        self.accuracy = RMSELoss()
        self.trafo_aug = []  # [AddNoise()]
        self.trafo = [ToTensor()]  # [SortSino(), ToTensor()]
        # self.example_input_array = [torch.empty(8, 1, 363, 180)]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        return {
            'optimizer': optimizer,
            'monitor': 'val_loss',
        }

    def forward(self, x):
        if self.hparams.residual:
            return x + self.net(x)
        return self.net(x)

    def training_step(self, batch, _):
        bilin_sino, full_sino, _ = batch
        znorm = ZNorm(bilin_sino)
        prediction = znorm.unnormalize(self(znorm.normalize(bilin_sino)))
        loss = self.loss(prediction, full_sino)
        return loss

    def validation_step(self, batch, batch_idx):
        bilin_sino, full_sino, fulldata = batch
        znorm = ZNorm(bilin_sino)
        prediction = znorm.unnormalize(self(znorm.normalize(bilin_sino)))
        loss = self.loss(prediction, full_sino)
        accuracy = self.accuracy(prediction, full_sino)

        # if self.current_epoch % 5 == 0 and batch_idx % 10 == 0:
        if batch_idx % 10 == 0:
            os.makedirs(
                pjoin(self.hparams.valid_dir, self.hparams.run_name,
                      f'{self.current_epoch}'),
                exist_ok=True,
            )
            img = fulldata.cpu().numpy()[0][0]
            cv2.imwrite(
                pjoin(self.hparams.valid_dir, self.hparams.run_name,
                      f'{self.current_epoch}/{batch_idx}_out_gt.png'),
                img/img.max()*255)

            reco = self.radon.backprojection(
                filter_sinogram(
                    prediction.transpose(-1, -2), 'hann')).cpu().numpy()[0][0]
            reco[reco < 0] = 0
            cv2.imwrite(
                pjoin(self.hparams.valid_dir, self.hparams.run_name,
                      f'{self.current_epoch}/{batch_idx}_out_pred.png'),
                reco/img.max()*255,
            )

        return {'val_loss': loss, 'accuracy': accuracy}

    def create_dataset(self, training: bool) -> SinoDataset:
        ds_path = pjoin(
            self.hparams.ds_dir,
            self.hparams.train_path if training else self.hparams.val_path)
        return SinoDataset(
            ds_path,
            transform=transforms.Compose(
                (self.trafo_aug + self.trafo) if training else self.trafo),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.create_dataset(training=True),
                          shuffle=True,
                          batch_size=self.hparams.batch_size,
                          pin_memory=False,
                          num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.create_dataset(training=False),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=False,
                          num_workers=0)

    def test_dataloader(self) -> DataLoader:
        pass

    def predict_dataloader(self) -> DataLoader:
        pass

    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('training', avg_loss)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('accuracy', avg_accuracy)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ds_dir', type=str)
        parser.add_argument('--train_path', type=str)
        parser.add_argument('--val_path', type=str)
        parser.add_argument('--valid_dir', type=str)
        parser.add_argument('--run_name', type=str)
        parser.add_argument('--subtype', type=str, default='sparse')
        parser.add_argument('--sparsity', type=int, default=8)
        parser.add_argument('--residual', type=bool)
        return parser


def main():
    # torch.set_num_threads(2)
    with open('config.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        ds_dir: str = json_data["ds_dir"]
        train_path: str = json_data["train_path"]
        val_path: str = json_data["val_path"]
        valid_dir: str = json_data["valid_dir"]
    subtype: str = 'sparse'
    sparsity: int = 4
    run_name: str = f'{subtype}{sparsity}_sino_unet'
    num_epochs: int = 151
    use_amp: bool = True

    parser = ArgumentParser()
    parser = SinoBilinUpsampling.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.lr = 1e-3
    hparams.max_epochs = num_epochs
    hparams.batch_size = 8
    hparams.ds_dir = ds_dir
    hparams.train_path = train_path.format(subtype=subtype,
                                           subnum=f'{sparsity}')
    hparams.val_path = val_path.format(subtype=subtype, subnum=f'{sparsity}')
    hparams.valid_dir = valid_dir
    hparams.run_name = run_name
    hparams.subtype = subtype
    hparams.sparsity = sparsity
    hparams.residual = True

    model = SinoBilinUpsampling(**vars(hparams))

    checkpoint_callback = ModelCheckpoint(
        dirpath=pjoin(valid_dir, run_name),
        monitor='val_loss',
        save_last=True,
    )

    trainer = Trainer(
        logger=True,
        precision=16 if use_amp else 32,
        gpus=1,
        callbacks=[checkpoint_callback],
        max_epochs=hparams.max_epochs,
        terminate_on_nan=True,
        deterministic=True,
        accumulate_grad_batches=4,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
