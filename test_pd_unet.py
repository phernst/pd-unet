import json
import os
from os.path import join as pjoin
from typing import Optional

import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_pd_unet import PrimalDual
from models.primal_dual import PrimalUnetBlock
from needle_simulations import generate_needle_simulations
from utilities.datasets import SinoDualDataset
from utilities.save_recons import validate_and_store, validate_consistency
from utilities.transforms import filter_sinogram, mu2hu

seed_everything(1701)


def get_iteration_weights(parfan: str,
                          subtype: str,
                          subnum: int):
    with open('config.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        valid_dir: str = json_data["valid_dir"]

    run_name: str = f'{subtype}{subnum}_pd_unet'

    checkpoint_dir = pjoin(valid_dir, 'img_per99_sino_znorm', parfan, run_name)
    checkpoint_path = sorted(
        [x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    out_dir = pjoin(
        "qualitative",
        f"{parfan}_{subtype}{subnum}",
        "pd_unet",
    )
    os.makedirs(out_dir, exist_ok=True)

    try:
        model = PrimalDual.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path))
    except AttributeError:
        model = PrimalDual.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), subtype=subtype,
            sparsity=subnum)
    assert isinstance(model.net.primal_blocks[0], PrimalUnetBlock), \
        "Make sure PrimalDualNetwork uses PrimalUnetBlocks"
    model.eval()

    return model.net.get_primal_dual_diff_weights()


def render_images_single_method(parfan: str,
                                subtype: str,
                                subnum: int):
    with open('config.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        ds_dir: str = f'ds_{parfan}'
        test_path: str = json_data["test_path"].format(subtype=subtype,
                                                       subnum=f'{subnum}')
        valid_dir: str = json_data["valid_dir"]

    run_name: str = f'{subtype}{subnum}_pd_unet'

    checkpoint_dir = pjoin(valid_dir, 'img_per99_sino_znorm', parfan, run_name)
    checkpoint_path = sorted(
        [x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    try:
        model = PrimalDual.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), parfan=parfan,)
    except AttributeError:
        model = PrimalDual.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), subtype=subtype,
            sparsity=subnum, parfan=parfan,)
    assert isinstance(model.net.primal_blocks[0], PrimalUnetBlock), \
        "Make sure PrimalDualNetwork uses PrimalUnetBlocks"

    model.eval()
    model.cuda()

    out_dir = pjoin(
        "qualitative",
        f"{parfan}_{subtype}{subnum}",
        "pd_unet",
    )
    os.makedirs(out_dir, exist_ok=True)

    dataloader_test = DataLoader(SinoDualDataset(
            pjoin(ds_dir, test_path),
            transform=Compose(model.trafo),
            return_meta=True,
        ),
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        num_workers=0)

    df = pd.read_csv(pjoin("qualitative", "preferred_slices.csv"))
    df = df[(df['parfan'] == parfan) &
            (df['stype'] == subtype) &
            (df['undersampling'] == subnum)]['idx']
    test_set_indices = [int(x) for x in df]
    print(test_set_indices)

    for idx, batch in enumerate(tqdm(dataloader_test)):
        if idx not in test_set_indices:
            continue

        with torch.inference_mode():
            batch, _, _, _, _, bilin_reco = batch
            _, _, under_reco, gt = batch

            sparse_sino = model.net.radon.forward(gt.float().cuda()).transpose(-1, -2)
            full_sino = model.radon.forward(gt.float().cuda()).transpose(-1, -2)

            full_sino = full_sino.cpu().numpy()

            prediction = model(sparse_sino.cuda(), under_reco.cuda())
            prediction_sino = model.radon.forward(prediction)
            prediction_sino = prediction_sino.transpose(-1, -2).cpu().numpy()
            prediction = prediction.cpu().numpy()
            prediction[prediction < 0] = 0

            gt = gt.numpy()
            gt[gt < 0] = 0

            bilin_reco = bilin_reco.numpy()
            bilin_reco[bilin_reco < 0] = 0

            under_reco = under_reco.numpy()
            under_reco[under_reco < 0] = 0

            np.save(pjoin(out_dir, f"{idx}_gt.npy"), mu2hu(gt[0, 0]))
            np.save(pjoin(out_dir, f"{idx}_pred.npy"), mu2hu(prediction[0, 0]))
            np.save(pjoin(out_dir, f"{idx}_sparse.npy"), mu2hu(under_reco[0, 0]))


def render_all_images():
    render_images_single_method("parallel", "sparse", 4)
    render_images_single_method("parallel", "sparse", 8)
    render_images_single_method("parallel", "sparse", 16)
    render_images_single_method("fan", "sparse", 4)
    render_images_single_method("fan", "sparse", 8)
    render_images_single_method("fan", "sparse", 16)


def render_needle_predictions(parfan: str,
                              subtype: str,
                              subnum: int,
                              groundtruth: torch.Tensor,
                              filename: str):
    with open('config.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        valid_dir: str = json_data["valid_dir"]

    run_name: str = f'{subtype}{subnum}_pd_unet'

    checkpoint_dir = pjoin(valid_dir, 'img_per99_sino_znorm', parfan, run_name)
    checkpoint_path = sorted(
        [x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    out_dir = pjoin(
        "qualitative_needle",
        f"{parfan}_{subtype}{subnum}",
    )
    os.makedirs(out_dir, exist_ok=True)

    try:
        model = PrimalDual.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), parfan=parfan,)
    except AttributeError:
        model = PrimalDual.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), subtype=subtype,
            sparsity=subnum, parfan=parfan,)
    assert isinstance(model.net.primal_blocks[0], PrimalUnetBlock), \
        "Make sure PrimalDualNetwork uses PrimalUnetBlocks"

    model.eval()
    model.cuda()

    gt = groundtruth

    with torch.inference_mode():
        sparse_sino = model.net.radon.forward(gt.float().cuda()).transpose(-1, -2)
        full_sino = model.radon.forward(gt.float().cuda()).transpose(-1, -2)
        under_reco = filter_sinogram(sparse_sino.transpose(-1, -2), 'hann')
        under_reco = model.net.radon.backprojection(under_reco)

        full_sino = full_sino.cpu().numpy()

        prediction = model(sparse_sino.cuda(), under_reco.cuda())
        prediction_sino = model.radon.forward(prediction)
        prediction_sino = prediction_sino.transpose(-1, -2).cpu().numpy()
        prediction = prediction.cpu().numpy()
        prediction[prediction < 0] = 0

    np.save(pjoin(out_dir, f'{filename}_pd_unet'), mu2hu(prediction))
    np.save(pjoin(out_dir, f'{filename}_under'), mu2hu(under_reco[0, 0].cpu().numpy()))
    np.save(pjoin(out_dir, f'{filename}_gt'), mu2hu(gt[0, 0].cpu().numpy()))


def main(parfan: str, subtype: str, subnum: int, update_wnb: Optional[bool] = False):
    with open('config.json', 'r') as json_file:
        json_data = json.load(json_file)
        ds_dir: str = f'ds_{parfan}'
        test_path: str = json_data["test_path"].format(subtype=subtype,
                                                       subnum=f'{subnum}')
        valid_dir: str = json_data["valid_dir"]
        out_dir: str = json_data["test_out_dir"]

    run_name: str = f'{subtype}{subnum}_pd_unet'

    checkpoint_dir = pjoin(valid_dir, 'img_per99_sino_znorm', parfan, run_name)
    checkpoint_path = sorted(
        [x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    os.makedirs(
        pjoin(out_dir, 'img_per99_sino_znorm', parfan, f'{subtype}{subnum}'),
        exist_ok=True)

    try:
        model = PrimalDual.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), parfan=parfan,)
    except AttributeError:
        model = PrimalDual.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), subtype=subtype,
            sparsity=subnum, parfan=parfan,)
    assert isinstance(model.net.primal_blocks[0], PrimalUnetBlock), \
        "Make sure PrimalDualNetwork uses PrimalUnetBlocks"
    model.eval()
    model.cuda()

    dataloader_test = DataLoader(SinoDualDataset(
            pjoin(ds_dir, test_path),
            transform=Compose(model.trafo),
            return_meta=True,
        ),
        shuffle=False,
        batch_size=model.hparams.batch_size,
        pin_memory=True,
        num_workers=model.hparams.batch_size)

    metrics = []
    for _, batch in enumerate(tqdm(dataloader_test)):
        with torch.inference_mode():
            batch, _, _, filename, subname, bilin_reco = batch
            _, _, under_reco, gt = batch

            sparse_sino = model.net.radon.forward(gt.float().cuda()).transpose(-1, -2)
            full_sino = model.radon.forward(gt.float().cuda()).transpose(-1, -2)

            full_sino = full_sino.cpu().numpy()

            prediction = model(sparse_sino.cuda(), under_reco.cuda())
            prediction_sino = model.radon.forward(prediction)
            prediction_sino = prediction_sino.transpose(-1, -2).cpu().numpy()
            prediction = prediction.cpu().numpy()
            prediction[prediction < 0] = 0

            gt = gt.numpy()
            gt[gt < 0] = 0

            bilin_reco = bilin_reco.numpy()
            bilin_reco[bilin_reco < 0] = 0

            for i, element in enumerate(bilin_reco):
                path4output = pjoin(out_dir, subname[i], filename[i])
                metrics.append(validate_and_store(
                    prediction[i].squeeze(),
                    gt[i].squeeze(),
                    element.squeeze(),
                    path4output, do_norm=False) | validate_consistency(
                        prediction_sino[i].squeeze(),
                        full_sino[i].squeeze(),
                    ))

    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(pjoin(out_dir, 'img_per99_sino_znorm', parfan,
                    f'{subtype}{subnum}', "Results_pd_unet.csv"))


if __name__ == '__main__':
    # render_all_images()
    # main('sparse', 4)
    render_needle_predictions('fan', 'sparse', 16, generate_needle_simulations('ABD_LYMPH_049.nii.gz'), 'ABD_LYMPH_049.nii.gz')
