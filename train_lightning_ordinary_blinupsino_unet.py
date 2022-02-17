import json
import os

from argparse import ArgumentParser
from os.path import join as pjoin
from typing import Any, List

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_radon import IRadon
from pytorch_radon.filters import HannFilter
from torch.utils.data import DataLoader
from torchvision import transforms

from losses.loss_type import LossType
from losses.restoration_loss import RestorationLoss
from losses.rmse_loss import RMSELoss
from models.unet import UNet as Net
from utilities.DS import MRISinoDataset
from utilities.transforms import ShiftCut, ToTensor, ZNorm

seed_everything(1701)

os.environ["CUDA_VISIBLE_DEVICES"]="2"


class SinoBilinUpsampling(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        fully_angles = np.linspace(0, kwargs['fullspokes'], num=kwargs['fullspokes'], endpoint=False) * kwargs['angle_inc']
        thetaSortInd = np.argsort(fully_angles % 360)
        self.iradon = IRadon(kwargs['imsize'], -fully_angles[thetaSortInd], circle=False, scikit=True, use_filter=HannFilter())
        self.net = Net()
        if self.hparams.loss == LossType.MSE:
            self.loss = torch.nn.MSELoss()
        elif self.hparams.loss == LossType.L1:
            self.loss = torch.nn.L1Loss()
        elif self.hparams.loss == LossType.RESTORATION:
            alpha = .02/(.02+.001)
            self.loss = RestorationLoss(alpha=alpha, data_range=kwargs['gtMax_sino'])
        self.accuracy = RMSELoss()
        self.trafo_aug = []  # [AddNoise()]
        self.trafo = [ShiftCut(imsize=kwargs['imsize'], PerformOnUnder=True, circle=False), ToTensor()]  # [SortSino(), ToTensor()]
        self.example_input_array = [torch.empty(8, 1, kwargs['sinosize'], fully_angles.shape[0])]

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
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        bilin_sino, full_sino, fulldata = batch
        znorm = ZNorm(bilin_sino)
        prediction = znorm.unnormalize(self(znorm.normalize(bilin_sino)))
        loss = self.loss(prediction, full_sino)
        accuracy = self.accuracy(prediction, full_sino)

        # if self.current_epoch % 5 == 0 and batch_idx % 10 == 0:
        # if batch_idx % 10 == 0:
        #     os.makedirs(
        #         pjoin(self.hparams.valid_dir, self.hparams.run_name,
        #               f'{self.current_epoch}'),
        #         exist_ok=True,
        #     )
        #     img = fulldata.cpu().numpy()[0][0]
        #     Image.fromarray((img/img.max()*255).astype(np.int8)).save(
        #               pjoin(self.hparams.valid_dir, self.hparams.run_name,
        #               f'{self.current_epoch}/{batch_idx}_out_gt.png'))

        #     reco = self.iradon(prediction).cpu().numpy()[0][0]
        #     reco[reco < 0] = 0
        #     Image.fromarray((reco/img.max()*255).astype(np.int8)).save(
        #               pjoin(self.hparams.valid_dir, self.hparams.run_name,
        #               f'{self.current_epoch}/{batch_idx}_out_pred.png'))

        return {'val_loss': loss, 'accuracy': accuracy}

    def create_dataset(self, training: bool) -> MRISinoDataset:
        return MRISinoDataset(
            self.hparams.train_path if training else self.hparams.val_path,
            transform=transforms.Compose(
                (self.trafo_aug + self.trafo) if training else self.trafo),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.create_dataset(training=True),
                          shuffle=True,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.create_dataset(training=False),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.workers)

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
        parser.add_argument('--train_path', type=str)
        parser.add_argument('--val_path', type=str)
        parser.add_argument('--valid_dir', type=str)
        parser.add_argument('--run_name', type=str)
        parser.add_argument('--residual', type=bool)
        parser.add_argument('--loss', type=LossType)
        return parser


def main(loss_type: LossType, jsonpath: str = 'config.json'):
    torch.set_num_threads(2)
    resume_from_checkpoint = True
    wnbactive = True
    num_epochs: int = 151
    use_amp: bool = True
    lr:float = 1e-3
    batch_size:int = 8
    accumulate_grad_batches: int = 4
    workers: int = 4

    with open(jsonpath, 'r') as json_file:
        json_data = json.load(json_file)
    
    run_name: str = "_".join([
                            json_data["run_prefix"], 
                            "GA" if json_data["isGA"] else "equidist", 
                            json_data["samplingtype"]+str(json_data["everyNSpoke"]) if json_data["samplingtype"]=="sparse" else json_data["samplingtype"]+str(json_data["firstNSpoke"]),
                            loss_type.name.lower(),
                            "sino_unet"
                            ])

    parser = ArgumentParser()
    parser = SinoBilinUpsampling.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.__dict__.update(json_data)
    hparams.lr = lr
    hparams.max_epochs = num_epochs
    hparams.batch_size = batch_size
    hparams.run_name = run_name
    hparams.residual = True
    hparams.loss = loss_type
    hparams.accumulate_grad_batches = accumulate_grad_batches
    hparams.use_amp = use_amp
    hparams.workers = workers
    hparams.angle_inc = 111.246117975 if json_data["isGA"] else (180/json_data["fullspokes"])

    if not wnbactive:
        os.environ["WANDB_MODE"] = "dryrun"

    model = SinoBilinUpsampling(**vars(hparams))

    checkpoint_callback = ModelCheckpoint(
        dirpath=pjoin(json_data["valid_dir"], run_name),
        monitor='val_loss',
        save_last=True,
    )
    logger = WandbLogger(name=run_name, id=run_name, project='SinoUp',
                            group='MRI', entity='ovgufindke', config=hparams)
    logger.watch(model, log='all', log_freq=100)

    if resume_from_checkpoint:
        chkpoint = pjoin(json_data["valid_dir"], run_name, "last.ckpt")
    else:
        chkpoint = None

    trainer = Trainer(
        logger=logger,
        precision=16 if use_amp else 32,
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        max_epochs=hparams.max_epochs,
        terminate_on_nan=True,
        deterministic=True,
        accumulate_grad_batches=accumulate_grad_batches,
        resume_from_checkpoint=chkpoint
    )
    trainer.fit(model)


if __name__ == '__main__':
    main(LossType.L1, 'config16.json')

    from test_lightning_ordinary_blinupsino_unet import main as TestMain
    TestMain(LossType.L1, 'config16.json')
