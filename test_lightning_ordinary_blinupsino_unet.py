import json
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from pytorch_radon import IRadon
from pytorch_radon.filters import HannFilter
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

import wandb
from losses.loss_type import LossType
from train_lightning_ordinary_blinupsino_unet import SinoBilinUpsampling
from utilities.DS import MRISinoDataset
from utilities.save_recons import ValidateNStore, save_sinograms
from utilities.transforms import ZNorm

seed_everything(1701)

os.environ["CUDA_VISIBLE_DEVICES"]="4"


def main(loss_type: LossType, jsonpath: str = 'config.json'):
    torch.set_num_threads(2)
    wnbactive = True 

    with open(jsonpath, 'r') as json_file:
        json_data = json.load(json_file)

    run_name: str = "_".join([
                        json_data["run_prefix"], 
                        "GA" if json_data["isGA"] else "equidist", 
                        json_data["samplingtype"]+str(json_data["everyNSpoke"]) if json_data["samplingtype"]=="sparse" else json_data["samplingtype"]+str(json_data["firstNSpoke"]),
                        loss_type.name.lower(),
                        "sino_unet"
                        ])

    checkpoint_dir = pjoin(json_data["valid_dir"], run_name)
    checkpoint_path = sorted([x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    out_dir = pjoin(json_data["test_out_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)

    model = SinoBilinUpsampling.load_from_checkpoint(
        pjoin(checkpoint_dir, checkpoint_path))
    model.eval()
    model.cuda()

    dataloader_test = DataLoader(MRISinoDataset(
            json_data["test_path"],
            transform=Compose(model.trafo),
            return_meta=True,
        ),
        shuffle=False,
        batch_size=model.hparams.batch_size,
        pin_memory=True, num_workers=model.hparams.workers)

    fully_angles = np.linspace(0, model.hparams.fullspokes, num=model.hparams.fullspokes, endpoint=False) * model.hparams.angle_inc
    thetaSortInd = np.argsort(fully_angles % 360)
    iradon_gpu = IRadon(model.hparams.imsize, -fully_angles[thetaSortInd], circle=False, use_filter=HannFilter(), scikit=True).cuda()

    if not wnbactive:
        os.environ["WANDB_MODE"] = "dryrun"

    with wandb.init(name=run_name, id=run_name, project="SinoUp", group="MRI", entity='ovgufindke', resume=True) as WnBRun:
        metrics = []
        for _, batch in enumerate(tqdm(dataloader_test)):
            with torch.no_grad():
                batch, _, _, filename, subname = batch
                bilin_sino, _, gt = batch

                bilin_sino = bilin_sino.cuda()
                znorm = ZNorm(bilin_sino)

                prediction_sino = znorm.unnormalize(model(znorm.normalize(bilin_sino)))
                prediction = iradon_gpu(prediction_sino).cpu().numpy()
                prediction[prediction < 0] = 0

                gt = gt.cpu().numpy()
                gt[gt < 0] = 0

                bilin_reco = iradon_gpu(bilin_sino).cpu().numpy()
                bilin_reco[bilin_reco < 0] = 0

                for i in range(len(bilin_reco)):
                    path4output = pjoin(out_dir, subname[i], filename[i])
                    save_sinograms(prediction_sino[i].squeeze().cpu().numpy(), bilin_sino[i].squeeze().cpu().numpy(), path4output)
                    metrics.append(ValidateNStore(prediction[i].squeeze(), gt[i].squeeze(), bilin_reco[i].squeeze(), path4output))

        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(pjoin(out_dir, f"Results_"+run_name+".csv"))
        WnBRun.summary["Test_PredSSIM"] = df["SSIM (Out)"].median()
        WnBRun.summary["Test_UnderSSIM"] = df["SSIM (Under)"].median()


if __name__ == '__main__':
    # main(LossType.MSE)
    main(LossType.L1)
    # main(LossType.RESTORATION)
