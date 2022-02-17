import numpy as np
import torch
import torch.nn.functional as F
from torch_radon.filtering import FourierFilters


class ToTensor:
    def __call__(self, sample):
        bilin_sino, full_sino, fulldata = sample
        bilin_sino = torch.from_numpy(bilin_sino).unsqueeze(0).float()
        full_sino = torch.from_numpy(full_sino).unsqueeze(0).float()
        fulldata = torch.from_numpy(fulldata).unsqueeze(0).float()
        return bilin_sino, full_sino, fulldata


class ToPrimalDualTensor:
    def __call__(self, sample):
        sparse_sino, full_sino, under_reco, fulldata = sample
        sparse_sino = torch.from_numpy(sparse_sino).unsqueeze(0).float()
        full_sino = torch.from_numpy(full_sino).unsqueeze(0).float()
        under_reco = torch.from_numpy(under_reco).unsqueeze(0).float()
        fulldata = torch.from_numpy(fulldata).unsqueeze(0).float()
        return sparse_sino, full_sino, under_reco, fulldata


class SortSino:
    def __init__(self, theta=None, n_spokes=512):
        if theta is None:
            theta = np.linspace(
                0, n_spokes, num=n_spokes, endpoint=False) * 111.246117975
        self.theta_sort_ind = np.argsort(theta % 360)
        # thetaSorted = theta[thetaSortInd]

    def __call__(self, sample):
        bilin_sino, full_sino, fulldata = sample
        full_sino = full_sino[:, self.theta_sort_ind]
        bilin_sino = bilin_sino[:, self.theta_sort_ind]
        return bilin_sino, full_sino, fulldata


class ZNorm:
    def __init__(self, tensor: torch.Tensor):
        self.mean = tensor.mean(dim=(2, 3), keepdims=True)
        self.std = tensor.std(dim=(2, 3), keepdims=True)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean)/self.std

    def unnormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean


def mu2hu(volume: torch.Tensor, mu_water: float = 0.02) -> torch.Tensor:
    return (volume - mu_water)/mu_water * 1000


def hu2mu(volume: np.ndarray, mu_water: float = 0.02) -> np.ndarray:
    return (volume * mu_water)/1000 + mu_water


def filter_sinogram(sinogram, filter_name="ramp"):
    size = sinogram.size(-1)
    n_angles = sinogram.size(-2)

    # Pad sinogram to improve accuracy
    padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
    pad = padded_size - size
    padded_sinogram = F.pad(sinogram.float(), (0, pad, 0, 0))

    sino_fft = torch.fft.rfft(padded_sinogram)

    # get filter and apply
    f = torch.from_numpy(FourierFilters.construct_fourier_filter(padded_size, filter_name)).to(sinogram.device)
    filtered_sino_fft = sino_fft * f

    # Inverse fft
    filtered_sinogram = torch.fft.irfft(filtered_sino_fft)
    filtered_sinogram = filtered_sinogram[..., :-pad] * (np.pi / (2 * n_angles))

    return filtered_sinogram.to(dtype=sinogram.dtype)
