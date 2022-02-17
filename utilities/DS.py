import copy
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


class MRISinoDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: str, 
                 return_interpolated: bool = True,
                 transform: Optional[Callable[..., Any]] = None,
                 return_meta: bool = False):
        self.dataset = h5py.File(root_path, 'r', libver='latest', swmr=True)
        self.transform = transform
        self.return_interpolated = return_interpolated
        self.return_meta = return_meta

    def __len__(self) -> int:
        """For returning the length of the file list"""
        return len(self.dataset)

    def Copy(self):
        """Creates a copy of the current object"""
        return copy.deepcopy(self)

    def __getitem__(self, idx):
        h5group = self.dataset[list(self.dataset.keys())[idx]]
        fulldata = h5group['imgMag'][()]
        fulldata[fulldata < 0] = 0

        if self.return_interpolated:
            under_sino = h5group['sinoMagUp'][()]
        else:
            under_sino = h5group['underSinoMag'][()]
        full_sino = h5group['sinoMag'][()]

        sample = (under_sino, full_sino, fulldata)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_meta:
            under_sino = h5group['underSinoMag'][()]
            underAngle = h5group['underAngle'][()]
            fileName = h5group['fileName'][()].astype(np.str)[0]
            subjectName = h5group['subjectName'][()].astype(np.str)[0]
            return sample, under_sino, underAngle, fileName, subjectName

        return sample


class MRISinoDualDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: str,
                 transform: Optional[Callable[..., Any]] = None,
                 return_meta: bool = False):
        self.dataset = h5py.File(root_path, 'r', libver='latest', swmr=True)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        """For returning the length of the file list"""
        return len(self.dataset)

    def Copy(self):
        """Creates a copy of the current object"""
        return copy.deepcopy(self)

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
            underAngle = h5group['underAngle'][()]
            fileName = h5group['fileName'][()].astype(np.str)[0]
            subjectName = h5group['subjectName'][()].astype(np.str)[0]
            underImgMagUp = h5group['underImgMagUp'][()]
            return sample, under_sino, underAngle, fileName, subjectName, underImgMagUp

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

    def Copy(self):
        """Creates a copy of the current object"""
        return copy.deepcopy(self)

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
            underAngle = h5group['underAngle'][()]
            fileName = h5group['fileName'][()].astype(np.str)[0]
            subjectName = h5group['subjectName'][()].astype(np.str)[0]
            return sample, under_sino, underAngle, fileName, subjectName

        return sample
