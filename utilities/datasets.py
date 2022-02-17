from enum import Enum, auto
from typing import Any, Callable, Optional, NamedTuple

import h5py
import numpy as np
import torch


class SpaceNormalization(NamedTuple):
    sino: float
    img: float


class DatasetType(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


class SinoDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: str,
                 transform: Optional[Callable[..., Any]] = None,
                 return_meta: bool = False):
        self.dataset = h5py.File(root_path, 'r', libver='latest', swmr=True)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        """For returning the length of the file list"""
        return len(self.dataset)

    def __getitem__(self, idx):
        h5group = self.dataset[list(self.dataset.keys())[idx]]
        fulldata = h5group['imgMag'][()]
        fulldata[fulldata < 0] = 0

        bilin_sino = h5group['sinoMagUp'][()]
        full_sino = h5group['sinoMag'][()]

        sample = (bilin_sino, full_sino, fulldata)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_meta:
            under_sino = h5group['underSinoMag'][()]
            under_angle = h5group['underAngle'][()]
            file_name = h5group['fileName'][()].astype(np.str)
            subject_name = h5group['subjectName'][()].astype(np.str)
            return sample, under_sino, under_angle, file_name, subject_name

        return sample


class SinoDualDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: str,
                 transform: Optional[Callable[..., Any]] = None,
                 return_meta: bool = False):
        self.dataset = h5py.File(root_path, 'r', libver='latest', swmr=True)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        """For returning the length of the file list"""
        return len(self.dataset)

    def __getitem__(self, idx):
        h5group = self.dataset[list(self.dataset.keys())[idx]]
        fulldata = h5group['imgMag'][()]
        fulldata[fulldata < 0] = 0

        sparse_sino = h5group['underSinoMag'][()]
        full_sino = h5group['sinoMag'][()]
        under_reco = h5group['underImgMag'][()]

        sample = (sparse_sino, full_sino, under_reco, fulldata)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_meta:
            under_sino = h5group['underSinoMag'][()]
            under_angle = h5group['underAngle'][()]
            file_name = h5group['fileName'][()].astype(np.str)
            subject_name = h5group['subjectName'][()].astype(np.str)
            under_img_mag_up = h5group['underImgMagUp'][()]
            return sample, under_sino, under_angle, file_name, subject_name, \
                under_img_mag_up

        return sample


class MRIRecoDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: str,
                 transform: Optional[Callable[..., Any]] = None,
                 return_meta: bool = False):
        self.dataset = h5py.File(root_path, 'r', libver='latest', swmr=True)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        """For returning the length of the file list"""
        return len(self.dataset)

    def __getitem__(self, idx):
        h5group = self.dataset[list(self.dataset.keys())[idx]]
        fulldata = h5group['imgMag'][()]
        fulldata[fulldata < 0] = 0

        sparse_img = h5group['underImgMag'][()]
        full_sino = h5group['sinoMag'][()]

        sample = (sparse_img, full_sino, fulldata)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_meta:
            under_sino = h5group['underSinoMag'][()]
            under_angle = h5group['underAngle'][()]
            file_name = h5group['fileName'][()].astype(np.str)
            subject_name = h5group['subjectName'][()].astype(np.str)
            return sample, under_sino, under_angle, file_name, subject_name

        return sample
