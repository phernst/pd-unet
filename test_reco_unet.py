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

from train_reco_unet import FBPConvNet
from utilities.datasets import MRIRecoDataset
from utilities.fan_geometry import default_fan_geometry
from utilities.save_recons import validate_and_store, validate_consistency
from utilities.transforms import mu2hu, filter_sinogram

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

    run_name: str = f'{subtype}{subnum}_reco_unet'

    checkpoint_dir = pjoin(valid_dir, 'img_per99_sino_znorm', parfan, run_name)
    checkpoint_path = sorted(
        [x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    out_dir = pjoin(
        "qualitative",
        f"{parfan}_{subtype}{subnum}",
        "reco_unet",
    )
    os.makedirs(out_dir, exist_ok=True)

    try:
        model = FBPConvNet.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), parfan=parfan,)
    except AttributeError:
        model = FBPConvNet.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), subtype=subtype,
            sparsity=subnum, parfan=parfan,)
    model.eval()
    model.cuda()

    dataloader_test = DataLoader(MRIRecoDataset(
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

    norm = model.hparams.per99.img
    for idx, batch in enumerate(tqdm(dataloader_test)):
        if idx not in test_set_indices:
            continue

        with torch.inference_mode():
            batch, _, _, _, _ = batch
            sparse_img, _, gt = batch

            full_sino = model.radon.forward(gt.float().cuda()).transpose(-1, -2)
            full_sino = full_sino.cpu().numpy()

            prediction = model(sparse_img.cuda()/norm)*norm
            sino = model.radon.forward(prediction).transpose(-1, -2)
            sino = sino.cpu().numpy()
            prediction = prediction.cpu().numpy()
            prediction[prediction < 0] = 0

            gt = gt.numpy()
            gt[gt < 0] = 0

            sparse_img = sparse_img.numpy()
            sparse_img[sparse_img < 0] = 0

            np.save(pjoin(out_dir, f"{idx}_gt.npy"), mu2hu(gt[0, 0]))
            np.save(pjoin(out_dir, f"{idx}_pred.npy"), mu2hu(prediction[0, 0]))
            np.save(pjoin(out_dir, f"{idx}_sparse.npy"), mu2hu(sparse_img[0, 0]))


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

    run_name: str = f'{subtype}{subnum}_reco_unet'

    checkpoint_dir = pjoin(valid_dir, 'img_per99_sino_znorm', parfan, run_name)
    checkpoint_path = sorted(
        [x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    try:
        model = FBPConvNet.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), parfan=parfan,)
    except AttributeError:
        model = FBPConvNet.load_from_checkpoint(
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
        under_reco = filter_sinogram(sparse_sino.transpose(-1, -2), 'hann')
        under_reco = radon.backprojection(under_reco)

        norm = model.hparams.per99.img
        prediction = model(under_reco.cuda()/norm)*norm
        sino = model.radon.forward(prediction).transpose(-1, -2)
        sino = sino.cpu().numpy()
        prediction = prediction.cpu().numpy()
        prediction[prediction < 0] = 0

    np.save(pjoin(out_dir, f'{filename}_reco_unet'), mu2hu(prediction))
    np.save(pjoin(out_dir, f'{filename}_under'), mu2hu(under_reco[0, 0].cpu().numpy()))


def main(parfan: str, subtype: str, subnum: int, update_wnb: Optional[bool] = False):
    with open('config.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        ds_dir: str = f'ds_{parfan}'
        test_path: str = json_data["test_path"].format(subtype=subtype,
                                                       subnum=f'{subnum}')
        valid_dir: str = json_data["valid_dir"]
        out_dir: str = json_data["test_out_dir"]

    run_name: str = f'{subtype}{subnum}_reco_unet'

    checkpoint_dir = pjoin(valid_dir, 'img_per99_sino_znorm', parfan, run_name)
    checkpoint_path = sorted(
        [x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    os.makedirs(
        pjoin(out_dir, 'img_per99_sino_znorm', parfan, f'{subtype}{subnum}'),
        exist_ok=True)

    try:
        model = FBPConvNet.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), parfan=parfan,)
    except AttributeError:
        model = FBPConvNet.load_from_checkpoint(
            pjoin(checkpoint_dir, checkpoint_path), subtype=subtype,
            sparsity=subnum, parfan=parfan,)
    model.eval()
    model.cuda()

    dataloader_test = DataLoader(MRIRecoDataset(
            pjoin(ds_dir, test_path),
            transform=Compose(model.trafo),
            return_meta=True,
        ),
        shuffle=False,
        batch_size=model.hparams.batch_size,
        pin_memory=True,
        num_workers=model.hparams.batch_size)

    metrics = []
    norm = model.hparams.per99.img
    for _, batch in enumerate(tqdm(dataloader_test)):
        with torch.inference_mode():
            batch, _, _, filename, subname = batch
            sparse_img, _, gt = batch

            full_sino = model.radon.forward(gt.float().cuda()).transpose(-1, -2)
            full_sino = full_sino.cpu().numpy()

            prediction = model(sparse_img.cuda()/norm)*norm
            sino = model.radon.forward(prediction).transpose(-1, -2)
            sino = sino.cpu().numpy()
            prediction = prediction.cpu().numpy()
            prediction[prediction < 0] = 0

            gt = gt.numpy()
            gt[gt < 0] = 0

            for i, element in enumerate(gt):
                path4output = pjoin(out_dir, subname[i], filename[i])
                metrics.append(validate_and_store(
                    prediction[i].squeeze(),
                    element.squeeze(),
                    np.random.rand(*gt[i].shape).squeeze(),  # TODO
                    path4output, do_norm=False) | validate_consistency(
                        sino[i].squeeze(),
                        full_sino[i].squeeze(),
                    ))

    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(pjoin(out_dir, 'img_per99_sino_znorm', parfan,
                    f'{subtype}{subnum}', "Results_reco_unet.csv"))


if __name__ == '__main__':
    render_all_images()
