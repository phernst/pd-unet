import numpy as np
import torch
from scipy.ndimage import zoom
from utilities.shiftcut import shift_cut_sinogram


class ToTensor(object):
    def __call__(_, sample):
        bilin_sino, full_sino, fulldata = sample
        bilin_sino = torch.from_numpy(bilin_sino).unsqueeze(0).float()
        full_sino = torch.from_numpy(full_sino).unsqueeze(0).float()
        fulldata = torch.from_numpy(fulldata).unsqueeze(0).float()
        return bilin_sino, full_sino, fulldata


class ToPrimalDualTensor(object):
    def __call__(_, sample):
        sparse_sino, full_sino, under_reco, fulldata = sample
        sparse_sino = torch.from_numpy(sparse_sino).unsqueeze(0).float()
        full_sino = torch.from_numpy(full_sino).unsqueeze(0).float()
        under_reco = torch.from_numpy(under_reco).unsqueeze(0).float()
        fulldata = torch.from_numpy(fulldata).unsqueeze(0).float()
        return sparse_sino, full_sino, under_reco, fulldata


class SortSino(object):
    def __init__(self, theta=None, nSpokes=512):
        if theta is None:
            theta = np.linspace(0, nSpokes, num=nSpokes, endpoint=False) * 111.246117975
        self.thetaSortInd = np.argsort(theta % 360)
        # thetaSorted = theta[thetaSortInd]

    def __call__(self, sample):
        bilin_sino, full_sino, fulldata = sample
        full_sino = full_sino[:, self.thetaSortInd]
        bilin_sino = bilin_sino[:, self.thetaSortInd]
        return bilin_sino, full_sino, fulldata

class FFTSinoResizer(object):
    def __init__(self, sinosize=None, PerformOnUnder=True):
        self.sinosize = sinosize
        self.PerformOnUnder = PerformOnUnder

    def __fftresize__(self, inp, start, end):
        return np.abs(
                    np.fft.ifft(
                        np.fft.ifftshift(
                            np.fft.fftshift(
                                np.fft.fft(inp, axis=1)
                            , axes=1)[start:-end]
                        , axes=1)
                    , axis=1)
                )

    def __call__(self, data):
        if self.sinosize is not None and data[0].shape[0]!=self.sinosize:
            start = (data[1].shape[0]-self.sinosize)//2
            end = (data[1].shape[0]-self.sinosize) - start
            if self.PerformOnUnder:
                d0 = self.__fftresize__(data[0],start,end)   #running on under   
            else:
                d0 = data[0]
            d1 = self.__fftresize__(data[1],start,end)   #running on fully
            data = (d0, d1, *data[2:])
        return data

class ShiftCut(object):
    def __init__(self, imsize=None, PerformOnUnder=True, circle=False):
        self.imsize = imsize
        self.circle = circle
        self.PerformOnUnder = PerformOnUnder

    def __clampzero__(self, data):
        return np.clip(data, a_min=0, a_max=data.max())

    def __call__(self, data):
        if self.imsize is not None:
            if self.PerformOnUnder:
                d0 = self.__clampzero__(shift_cut_sinogram(data[0], img_size=self.imsize, circle=self.circle))   #running on under      
            else:
                d0 = data[0]
            d1 = self.__clampzero__(shift_cut_sinogram(data[1], img_size=self.imsize, circle=self.circle))   #running on fully
            data = (d0, d1, *data[2:])
        return data

class SinoUpsample(object):
    def __init__(self, fully_angle, under_angle):
        self.fully_angle = fully_angle
        self.under_angle = under_angle
        self.finalloc_underangle = np.where(self.fully_angle==self.under_angle[-1])[0][0]

    def __call__(self, data):
        assert data[0].shape[0] % 2 == 1, "Only implemented for odd shapes"
        first_up = zoom(
            data[0],
            # (1., (self.under_angle[-1] + 1)/len(self.under_angle)),
            (1., (self.finalloc_underangle+1)/data[0].shape[1]),
            order=1,
            prefilter=False)
        last_part = np.concatenate([
            data[0][:, -1:],
            data[0][::-1, :1]
            ], axis=1)
        last_up = zoom(
            last_part,
            # (1, (180-self.under_angle[-1]+1)/2),
            (1, ((len(self.fully_angle)-self.finalloc_underangle)+1)/2),
            order=1,
            prefilter=False)[:, 1:-1]
        return (np.concatenate([first_up, last_up], axis=1), *data[1:])

class ZNorm(object):
    def __init__(self, tensor: torch.tensor):
        self.mean = tensor.mean(dim=(2, 3), keepdims=True)
        self.std = tensor.std(dim=(2, 3), keepdims=True)

    def normalize(self, tensor: torch.tensor) -> torch.tensor:
        return (tensor - self.mean)/self.std

    def unnormalize(self, tensor: torch.tensor) -> torch.tensor:
        return tensor * self.std + self.mean
