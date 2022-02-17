import json
from typing import Tuple
import os
from os.path import join as pjoin

import h5py
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
import torch
from torch_radon import RadonFanbeam
from tqdm import tqdm

from utilities.datasets import DatasetType
from utilities.fan_geometry import default_fan_geometry


def fan_radon(data_slice: np.ndarray, theta: np.ndarray, full: bool) -> np.ndarray:
    if not hasattr(fan_radon, "full_radon") or not hasattr(fan_radon, "reduced_radon"):
        fan_geom = default_fan_geometry()
        fan_radon.full_radon = RadonFanbeam(
            resolution=data_slice.shape[0],
            angles=np.deg2rad(np.arange(360)),
            source_distance=fan_geom.source_distance,
            det_distance=fan_geom.det_distance,
            det_count=fan_geom.det_count,
            det_spacing=fan_geom.det_spacing,
        )
        fan_radon.reduced_radon = RadonFanbeam(
            resolution=data_slice.shape[0],
            angles=np.deg2rad(theta),
            source_distance=fan_geom.source_distance,
            det_distance=fan_geom.det_distance,
            det_count=fan_geom.det_count,
            det_spacing=fan_geom.det_spacing,
        )
    if len(theta) != len(fan_radon.reduced_radon.angles):
        fan_radon.reduced_radon = RadonFanbeam(
            resolution=data_slice.shape[0],
            angles=np.deg2rad(theta),
            source_distance=fan_geom.source_distance,
            det_distance=fan_geom.det_distance,
            det_count=fan_geom.det_count,
            det_spacing=fan_geom.det_spacing,
        )

    data_slice_t = torch.from_numpy(data_slice).float().cuda()
    data_slice_t.unsqueeze_(0)
    sino_t = (fan_radon.full_radon if full else fan_radon.reduced_radon).forward(data_slice_t)
    sino = sino_t[0].transpose(0, 1).cpu().numpy()
    return sino


def fan_iradon(sino: np.ndarray, theta: np.ndarray, full: bool) -> np.ndarray:
    if fan_radon.full_radon is None or fan_radon.reduced_radon is None:
        raise NotImplementedError("Please call `fan_radon` first.")
    if len(fan_radon.reduced_radon.angles) != len(theta):
        raise NotImplementedError("Wrong angles. Call `fan_radon` again.")

    sino_t = torch.from_numpy(sino).float().cuda().transpose(0, 1)
    sino_t.unsqueeze_(0)
    radon = fan_radon.full_radon if full else fan_radon.reduced_radon
    reco_t = radon.backprojection(radon.filter_sinogram(sino_t, 'hann'))
    reco = reco_t[0].cpu().numpy()
    return reco


# creates a fully sampled sinogram and its reco
def generate_sino(data_slice: np.ndarray, every_nth_view: int) -> np.ndarray:
    theta = np.arange(360)[::every_nth_view]
    sino = fan_radon(data_slice, theta=theta, full=True)
    img = fan_iradon(sino, theta=theta, full=True)
    return sino, img


def generate_sino_limited(data_slice: np.ndarray, first_n_spokes: int) -> np.ndarray:
    theta = np.arange(360)[:first_n_spokes]
    sino = fan_radon(data_slice, theta=theta, full=True)
    img = fan_iradon(sino, theta=theta, full=True)
    return sino, img


def undersample_sino(full_sino: np.ndarray, every_nth_view: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    under_sino = full_sino[:, ::every_nth_view]
    under_angle = np.arange(360)[::every_nth_view]
    under_img = fan_iradon(under_sino, theta=under_angle, full=False)
    return under_sino, under_angle, under_img


def undersample_sino_limited(full_sino: np.ndarray, first_n_spokes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    under_sino = full_sino[:, :first_n_spokes]
    under_angle = np.arange(360)[:first_n_spokes]
    under_img = fan_iradon(under_sino, theta=under_angle, full=False)
    return under_sino, under_angle, under_img


def custom_upsample(under_sino: np.ndarray, every_nth_view: int) -> np.ndarray:
    assert under_sino.shape[0] % 2 == 1, "Only implemented for odd shapes"
    under_angle = np.arange(360)[::every_nth_view]
    first_up = zoom(
        under_sino,
        (1., (under_angle[-1] + 1)/len(under_angle)),
        order=1,
        prefilter=False)
    last_part = np.concatenate([
        under_sino[:, -1:],
        under_sino[:, :1]
        ], axis=1)
    last_up = zoom(
        last_part,
        (1, (360-under_angle[-1]+1)/2),
        order=1,
        prefilter=False)[:, 1:-1]
    return np.concatenate([first_up, last_up], axis=1)


def custom_upsample_limited(under_sino: np.ndarray, first_n_spokes: int) -> np.ndarray:
    assert under_sino.shape[0] % 2 == 1, "Only implemented for odd shapes"
    under_angle = np.arange(360)[:first_n_spokes]
    last_part = np.concatenate([
        under_sino[:, -1:],
        under_sino[:, :1]
        ], axis=1)
    last_up = zoom(
        last_part,
        (1, (360-under_angle[-1]+1)/2),
        order=1,
        prefilter=False)[:, 1:-1]
    return np.concatenate([under_sino, last_up], axis=1)


def upsample_sino(under_sino: np.ndarray, every_nth_view: int) -> Tuple[np.ndarray, np.ndarray]:
    sino_up = custom_upsample(under_sino, every_nth_view)
    under_image_up = fan_iradon(sino_up,
                                theta=np.arange(360)[::every_nth_view],
                                full=True)
    return sino_up, under_image_up


def upsample_sino_limited(under_sino: np.ndarray, first_n_spokes: int) -> Tuple[np.ndarray, np.ndarray]:
    sino_up = custom_upsample_limited(under_sino, first_n_spokes)
    under_image_up = fan_iradon(sino_up,
                                theta=np.arange(360)[:first_n_spokes],
                                full=True)
    return sino_up, under_image_up


def process_slice_sparse(filename: str,
                         idx_z: int,
                         every_nth_view: int,
                         data_slice,
                         h5) -> None:
    h5ds = h5.create_group(f"{filename.split('.')[0]}_{idx_z}")
    h5ds.create_dataset(
        'fileName', data=np.ndarray(filename).astype('S'))
    h5ds.create_dataset('subjectName', data=np.ndarray(
        filename.split('.')[0]).astype('S'))

    data_slice = resize(data_slice, (256, 256))
    sino, img = generate_sino(data_slice, every_nth_view)
    under_sino, under_angle, under_img = undersample_sino(
        sino, every_nth_view)
    sino_up, under_img_up = upsample_sino(
        under_sino, every_nth_view)

    h5ds.create_dataset('sinoMag', data=sino)
    h5ds.create_dataset('imgMag', data=img)
    h5ds.create_dataset('underSinoMag', data=under_sino)
    h5ds.create_dataset('underImgMag', data=under_img)
    h5ds.create_dataset('underAngle', data=under_angle)
    h5ds.create_dataset('sinoMagUp', data=sino_up)
    h5ds.create_dataset('underImgMagUp', data=under_img_up)


def process_slice_limited(filename: str,
                          idx_z: int,
                          first_n_spokes: int,
                          data_slice,
                          h5) -> None:
    h5ds = h5.create_group(f"{filename.split('.')[0]}_{idx_z}")
    h5ds.create_dataset(
        'fileName', data=np.ndarray(filename).astype('S'))
    h5ds.create_dataset('subjectName', data=np.ndarray(
        filename.split('.')[0]).astype('S'))

    data_slice = resize(data_slice, (256, 256))
    sino, img = generate_sino_limited(data_slice, first_n_spokes)
    under_sino, under_angle, under_img = undersample_sino_limited(
        sino, first_n_spokes)
    sino_up, under_img_up = upsample_sino_limited(
        under_sino, first_n_spokes)

    h5ds.create_dataset('sinoMag', data=sino)
    h5ds.create_dataset('imgMag', data=img)
    h5ds.create_dataset('underSinoMag', data=under_sino)
    h5ds.create_dataset('underImgMag', data=under_img)
    h5ds.create_dataset('underAngle', data=under_angle)
    h5ds.create_dataset('sinoMagUp', data=sino_up)
    h5ds.create_dataset('underImgMagUp', data=under_img_up)


def test_custom_upsample():
    img = np.zeros((256, 256), dtype=float)
    img[120:140, 120:140] = 1
    img[141:150, 141:150] = 0.5
    every_nth_view = 16
    theta = np.arange(360)[::every_nth_view]
    sino = fan_radon(img, theta=theta, full=True)
    sp_sino = sino[:, ::every_nth_view]
    up_sino = custom_upsample(sp_sino, every_nth_view)
    resize_sino = resize(sp_sino, sino.shape, order=1)
    from matplotlib import pyplot as plt
    plt.imshow(np.abs(sino - up_sino))
    plt.figure()
    plt.imshow(np.abs(sino - resize_sino))
    plt.figure()
    plt.imshow(fan_iradon(up_sino, theta, full=True))
    plt.figure()
    plt.imshow(fan_iradon(resize_sino, theta, full=True))
    plt.show()
    print(f'new abs diff: {np.mean(np.abs(sino - up_sino))}')
    print(f'old abs diff: {np.mean(np.abs(sino - resize_sino))}')


def create_sparse_datasets(every_nth_view: int, ds_type: DatasetType):
    data_path = 'big_data'

    with open('config.json', 'r', encoding='utf-8') as json_file:
        json_dict = json.load(json_file)
        ds_dir: str = json_dict["ds_dir"]

    # total num slices (train + valid) = 4185
    with open('train_valid.json', 'r', encoding='utf-8') as json_file:
        json_dict = json.load(json_file)
        num_train_datasets: int = len(json_dict['train_files'])
        num_valid_datasets: int = len(json_dict['valid_files'])
    num_slices_per_file = int(4185/(num_train_datasets + num_valid_datasets))

    ds_type_str = ds_type.name.lower()
    output_file = f'sparse{every_nth_view}_{ds_type_str}.h5'

    only_files = []
    with open('train_valid.json', 'r', encoding='utf-8') as json_file:
        only_files += json.load(json_file)[f'{ds_type_str}_files']
    # if len(only_files) == 0:
    #     only_files = data_filenames

    h5 = h5py.File(pjoin(ds_dir, output_file), "w")
    with tqdm(total=num_slices_per_file*len(only_files)) as pbar:
        for filename in only_files:
            nib_data = nib.load(os.path.join(data_path, filename)).get_fdata()
            z_range = [int((nib_data.shape[-1]-num_slices_per_file) //
                       2) + idx for idx in range(num_slices_per_file)]
            for idx_z in z_range:
                process_slice_sparse(filename, idx_z, every_nth_view,
                                     nib_data[..., idx_z].transpose(), h5)
                pbar.update(1)


def create_limited_datasets(first_n_spokes: int, ds_type: DatasetType):
    data_path = 'big_data'

    with open('config.json', 'r', encoding='utf-8') as json_file:
        json_dict = json.load(json_file)
        ds_dir: str = json_dict["ds_dir"]

    # total num slices (train + valid) = 4185
    with open('train_valid.json', 'r', encoding='utf-8') as json_file:
        json_dict = json.load(json_file)
        num_train_datasets: int = len(json_dict['train_files'])
        num_valid_datasets: int = len(json_dict['valid_files'])
    num_slices_per_file = int(4185/(num_train_datasets + num_valid_datasets))

    ds_type_str = ds_type.name.lower()
    output_file = f'limited{first_n_spokes}_{ds_type_str}.h5'

    only_files = []
    with open('train_valid.json', 'r', encoding='utf-8') as json_file:
        only_files += json.load(json_file)[f'{ds_type_str}_files']

    h5 = h5py.File(pjoin(ds_dir, output_file), "w")
    with tqdm(total=num_slices_per_file*len(only_files)) as pbar:
        for filename in only_files:
            nib_data = nib.load(os.path.join(data_path, filename)).get_fdata()
            z_range = [int((nib_data.shape[-1]-num_slices_per_file) //
                       2) + idx for idx in range(num_slices_per_file)]
            for idx_z in z_range:
                process_slice_limited(filename, idx_z, first_n_spokes,
                                      nib_data[..., idx_z].transpose(), h5)
                pbar.update(1)


if __name__ == '__main__':
    create_sparse_datasets(16, DatasetType.TRAIN)
    create_sparse_datasets(16, DatasetType.VALID)
    create_sparse_datasets(16, DatasetType.TEST)
