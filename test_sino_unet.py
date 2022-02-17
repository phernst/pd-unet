import json
import os
from os.path import join as pjoin
from typing import Optional

import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
import torch
from torch_radon.radon import RadonFanbeam
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from tqdm import tqdm
from createh5_ct import custom_upsample

from train_sino_unet import SinoBilinUpsampling
from utilities.datasets import SinoDataset
from utilities.fan_geometry import default_fan_geometry
from utilities.save_recons import validate_and_store, save_sinograms
from utilities.save_recons import validate_consistency
from utilities.transforms import ZNorm, mu2hu, filter_sinogram

seed_everything(1701)


def render_images_single_method(parfan: str,
                                subtype: str,
                                subnum: int):
    with open('config.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        ds_dir: str = f'ds_{parfan}'
        test_path: str = json_data["test_path"].format(subtype=subtype,
                                                       subnum=f'{subnum}')
        valid_dir: str = json_data["valid_dir"]

    run_name: str = f'{subtype}{subnum}_sino_unet'

    checkpoint_dir = pjoin(valid_dir, 'img_per99_sino_znorm', parfan, run_name)
    checkpoint_path = sorted(
        [x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    out_dir = pjoin(
        "qualitative",
        f"{parfan}_{subtype}{subnum}",
        "sino_unet",
    )
    os.makedirs(out_dir, exist_ok=True)

    try:
        model = SinoBilinUpsampling.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), parfan=parfan,)
    except AttributeError:
        model = SinoBilinUpsampling.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), subtype=subtype,
            sparsity=subnum, parfan=parfan,)
    model.eval()
    model.cuda()

    dataloader_test = DataLoader(SinoDataset(
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
            batch, _, _, _, _ = batch
            bilin_sino, full_sino, gt = batch
            bilin_sino = bilin_sino.cuda()
            znorm = ZNorm(bilin_sino)

            full_sino = full_sino.cpu().numpy()

            norm_bilin_sino = znorm.normalize(bilin_sino)
            prediction_sino = znorm.unnormalize(model(norm_bilin_sino))
            prediction = model.radon.backprojection(
                filter_sinogram(
                    prediction_sino.transpose(-1, -2), 'hann')).cpu().numpy()
            prediction[prediction < 0] = 0
            prediction_sino = prediction_sino.cpu().numpy()

            gt = gt.numpy()
            gt[gt < 0] = 0

            bilin_reco = model.radon.backprojection(
                filter_sinogram(
                    bilin_sino.transpose(-1, -2), 'hann')).cpu().numpy()
            bilin_reco[bilin_reco < 0] = 0

            np.save(pjoin(out_dir, f"{idx}_gt.npy"), mu2hu(gt[0, 0]))
            np.save(pjoin(out_dir, f"{idx}_pred.npy"), mu2hu(prediction[0, 0]))


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
        out_dir: str = json_data["test_out_dir"]

    run_name: str = f'{subtype}{subnum}_sino_unet'

    checkpoint_dir = pjoin(valid_dir, 'img_per99_sino_znorm', parfan, run_name)
    checkpoint_path = sorted(
        [x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    try:
        model = SinoBilinUpsampling.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), parfan=parfan,)
    except AttributeError:
        model = SinoBilinUpsampling.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), subtype=subtype,
            sparsity=subnum, parfan=parfan,)
    model.eval()
    model.cuda()

    out_dir = pjoin(
        "qualitative_needle",
        f"{parfan}_{subtype}{subnum}",
    )
    os.makedirs(out_dir, exist_ok=True)

    gt = groundtruth

    geom = default_fan_geometry()
    radon = RadonFanbeam(
        256, np.deg2rad(np.arange(360))[::subnum],
        source_distance=geom.source_distance,
        det_distance=geom.det_distance,
        det_count=geom.det_count,
        det_spacing=geom.det_spacing,
    )

    with torch.inference_mode():
        sparse_sino = radon.forward(gt.float().cuda()).transpose(-1, -2)
        bilin_sino = torch.from_numpy(custom_upsample(
            sparse_sino[0, 0].cpu().numpy(),
            subnum))[None, None].float().cuda()

        znorm = ZNorm(bilin_sino)

        norm_bilin_sino = znorm.normalize(bilin_sino)
        prediction_sino = znorm.unnormalize(model(norm_bilin_sino))
        prediction = model.radon.backprojection(
            filter_sinogram(
                prediction_sino.transpose(-1, -2), 'hann')).cpu().numpy()
        prediction[prediction < 0] = 0
        prediction_sino = prediction_sino.cpu().numpy()

        bilin_reco = model.radon.backprojection(
            filter_sinogram(
                bilin_sino.transpose(-1, -2), 'hann'))
        bilin_reco[bilin_reco < 0] = 0

    np.save(pjoin(out_dir, f'{filename}_sino_unet'), mu2hu(prediction))
    np.save(pjoin(out_dir, f'{filename}_bilin_reco'), mu2hu(bilin_reco[0, 0].cpu().numpy()))


def main(parfan: str, subtype: str, subnum: int, update_wnb: Optional[bool] = False):
    with open('config.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        ds_dir: str = f'ds_{parfan}'
        test_path: str = json_data["test_path"].format(subtype=subtype,
                                                       subnum=f'{subnum}')
        valid_dir: str = json_data["valid_dir"]
        out_dir: str = json_data["test_out_dir"]

    run_name: str = f'{subtype}{subnum}_sino_unet'

    checkpoint_dir = pjoin(valid_dir, 'img_per99_sino_znorm', parfan, run_name)
    checkpoint_path = sorted(
        [x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    os.makedirs(
        pjoin(out_dir, 'img_per99_sino_znorm', parfan, f'{subtype}{subnum}'),
        exist_ok=True)

    try:
        model = SinoBilinUpsampling.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), parfan=parfan,)
    except AttributeError:
        model = SinoBilinUpsampling.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), subtype=subtype,
            sparsity=subnum, parfan=parfan,)
    model.eval()
    model.cuda()

    dataloader_test = DataLoader(SinoDataset(
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
            batch, _, _, filename, subname = batch
            bilin_sino, full_sino, gt = batch
            bilin_sino = bilin_sino.cuda()
            znorm = ZNorm(bilin_sino)

            full_sino = full_sino.cpu().numpy()

            norm_bilin_sino = znorm.normalize(bilin_sino)
            prediction_sino = znorm.unnormalize(model(norm_bilin_sino))
            prediction = model.radon.backprojection(
                filter_sinogram(
                    prediction_sino.transpose(-1, -2), 'hann')).cpu().numpy()
            prediction[prediction < 0] = 0
            prediction_sino = prediction_sino.cpu().numpy()

            gt = gt.numpy()
            gt[gt < 0] = 0

            bilin_reco = model.radon.backprojection(
                filter_sinogram(
                    bilin_sino.transpose(-1, -2), 'hann')).cpu().numpy()
            bilin_reco[bilin_reco < 0] = 0

            for i, element in enumerate(bilin_reco):
                path4output = pjoin(out_dir, subname[i], filename[i])
                save_sinograms(prediction_sino[i].squeeze(),
                               bilin_sino[i].squeeze().cpu().numpy(),
                               path4output)
                metrics.append(validate_and_store(
                    prediction[i].squeeze(),
                    gt[i].squeeze(),
                    element.squeeze(),
                    path4output, do_norm=False) | validate_consistency(
                        prediction_sino[i].squeeze(),
                        full_sino[i].squeeze(),
                        bilin_sino[i].squeeze().cpu().numpy(),
                    ))

    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(pjoin(out_dir, 'img_per99_sino_znorm', parfan,
                    f'{subtype}{subnum}', "Results_sino_unet.csv"))


if __name__ == '__main__':
    render_all_images()
