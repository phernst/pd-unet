import json
import os
from os.path import join as pjoin

import pandas as pd
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

import wandb
from losses.loss_type import LossType
from models.primal_dual import PrimalUnetBlock
from train_lightning_pd_unet import PrimalDual
from utilities.DS import MRISinoDualDataset
from utilities.save_recons import ValidateNStore

seed_everything(1701)

os.environ["CUDA_VISIBLE_DEVICES"]="1"


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
                        "pd_unet"
                        ])

    checkpoint_dir = pjoin(json_data["valid_dir"], run_name)
    checkpoint_path = sorted(
        [x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    out_dir = pjoin(json_data["test_out_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)

    model = PrimalDual.load_from_checkpoint(
        pjoin(checkpoint_dir, checkpoint_path))
    assert isinstance(model.net.primal_blocks[0], PrimalUnetBlock), \
        "Make sure PrimalDualNetwork uses PrimalUnetBlocks"
    model.eval()
    model.cuda()

    dataloader_test = DataLoader(MRISinoDualDataset(
            json_data["test_path"],
            transform=Compose(model.trafo),
            return_meta=True,
        ),
        shuffle=False,
        batch_size=model.hparams.batch_size,
        pin_memory=True, num_workers=model.hparams.workers)

    if not wnbactive:
        os.environ["WANDB_MODE"] = "dryrun"

    with wandb.init(name=run_name, id=run_name, project="SinoUp", group="MRI", entity='ovgufindke', resume=True) as WnBRun:
        metrics = []
        for _, batch in enumerate(tqdm(dataloader_test)):
            with torch.no_grad():
                batch, _, _, filename, subname, bilin_reco = batch
                sparse_sino, _, under_reco, gt = batch

                prediction = model(sparse_sino.cuda(), under_reco.cuda())
                prediction = prediction.cpu().numpy()
                prediction[prediction < 0] = 0

                gt = gt.cpu().numpy()
                gt[gt < 0] = 0

                bilin_reco = bilin_reco.cpu().numpy()
                bilin_reco[bilin_reco < 0] = 0

                for i in range(len(bilin_reco)):
                    path4output = pjoin(out_dir, subname[i], filename[i])
                    metrics.append(ValidateNStore(
                        prediction[i].squeeze(),
                        gt[i].squeeze(),
                        bilin_reco[i].squeeze(),
                        path4output))

        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(pjoin(out_dir, f"Results_"+run_name+".csv"))
        WnBRun.summary["Test_PredSSIM"] = df["SSIM (Out)"].median()
        WnBRun.summary["Test_UnderSSIM"] = df["SSIM (Under)"].median()


if __name__ == '__main__':
    # main(LossType.MSE)
    main(LossType.L1)
    # main(LossType.RESTORATION)
