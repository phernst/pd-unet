from typing import Optional, Union, List, Tuple

import numpy as np
from pytorch_radon import Radon, IRadon
from pytorch_radon.filters import HannFilter
import torch
from torch import nn
import torch.nn.functional as F

from utilities.transforms import ZNorm
from utilities.DS import SpaceNormalization
from .unet import UNet


class PrePadDualConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[List[int], Tuple[int, int], int]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        with torch.no_grad():
            self.conv.bias.zero_()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def _angle_padding(self, inp):
        return F.pad(
            inp, (self.kernel_size[1]//2, self.kernel_size[1]//2, 0, 0), mode='circular')

    def _proj_padding(self, inp):
        return F.pad(
            inp, (0, 0, self.kernel_size[0]//2, self.kernel_size[0]//2), mode='replicate')

    def forward(self, x):
        return self.conv(self._proj_padding(self._angle_padding(x)))


class PrePadPrimalConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[List[int], Tuple[int, int], int]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        with torch.no_grad():
            self.conv.bias.zero_()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def _lr_padding(self, inp: torch.tensor):
        return F.pad(
            inp, (self.kernel_size[1]//2, self.kernel_size[1]//2, 0, 0), mode='replicate')

    def _tb_padding(self, inp: torch.tensor):
        return F.pad(
            inp, (0, 0, self.kernel_size[0]//2, self.kernel_size[0]//2), mode='replicate')

    def forward(self, x: torch.tensor):
        return self.conv(self._tb_padding(self._lr_padding(x)))


class DualBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.layers = nn.Sequential(
            PrePadDualConv2D(features+2, 32, 3),
            nn.PReLU(32),
            PrePadDualConv2D(32, 32, 3),
            nn.PReLU(32),
            PrePadDualConv2D(32, features, 3),
        )
        self.diff_weight = nn.Parameter(torch.ones(1, features, 1, 1))

    def forward(self, h: torch.tensor, f: torch.tensor, g: torch.tensor):
        B, _, H, W = h.shape
        block_input = torch.cat([h, torch.mean(f, dim=1, keepdim=True), g], 1)
        return h + self.diff_weight.repeat(B, 1, H, W)*self.layers(block_input)


class PrimalBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.layers = nn.Sequential(
            PrePadPrimalConv2D(features+1, 32, 3),
            nn.PReLU(32),
            PrePadPrimalConv2D(32, 32, 3),
            nn.PReLU(32),
            PrePadPrimalConv2D(32, features, 3),
        )
        self.diff_weight = nn.Parameter(torch.zeros(1, features, 1, 1))

    def forward(self, h: torch.tensor, f: torch.tensor):
        B, _, H, W = f.shape
        block_input = torch.cat([f, torch.mean(h, dim=1, keepdim=True)], 1)
        return f + self.diff_weight.repeat(B, 1, H, W)*self.layers(block_input)


class PrimalUnetBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.layers = UNet(features+1, features, wf=5)
        self.diff_weight = nn.Parameter(torch.zeros(1, features, 1, 1))

    def forward(self, h, f):
        B, _, H, W = f.shape
        block_input = torch.cat([f, torch.mean(h, dim=1, keepdim=True)], 1)
        return f + self.diff_weight.repeat(B, 1, H, W)*self.layers(block_input)


class PrimalDualNetwork(nn.Module):
    def __init__(self, in_size: int,
                 theta: Optional[Union[List[float], np.array, torch.tensor]],
                 n_primary: int, n_dual: int, n_iterations: int,
                 use_original_block: bool = True,
                 use_original_init: bool = True,
                 norm: Optional[SpaceNormalization] = None):
        super().__init__()
        self.primal_blocks = nn.ModuleList([
            (PrimalBlock if use_original_block else PrimalUnetBlock)(n_primary)
            for _ in range(n_iterations)
        ])
        self.dual_blocks = nn.ModuleList([
            DualBlock(n_dual) for _ in range(n_iterations)
        ])
        self.radon = Radon(in_size, theta, circle=False, scikit=True)
        self.iradon = IRadon(in_size, theta, circle=False,
                             use_filter=HannFilter(), scikit=True)
        self.in_size = in_size
        self.n_primary = n_primary
        self.n_dual = n_dual
        self.use_original_init = use_original_init
        self.norm = norm or SpaceNormalization(1.0, 1.0)

    def get_primal_dual_diff_weights(self):
        return {
            'primal': [f.diff_weight.mean().item() for f in self.primal_blocks],
            'dual': [f.diff_weight.mean().item() for f in self.dual_blocks]
        }

    def forward(self, sino: torch.tensor, sparse_reco: torch.tensor,
                output_stages: bool = False):
        # g, h: z-normed
        # f: per99-normed
        g = sino
        znorm = ZNorm(g)
        g = znorm.normalize(g)
        B, _, P, A = g.shape
        h = torch.zeros(B, self.n_dual, P, A, device=g.device)
        if self.use_original_init:
            f = torch.zeros(B, self.n_primary, self.in_size,
                            self.in_size, device=g.device)
        else:
            f = sparse_reco.repeat(1, self.n_primary, 1, 1)/self.norm.img

        stages = []
        for primary_block, dual_block in zip(self.primal_blocks, self.dual_blocks):
            h = dual_block(h, znorm.normalize(self.radon(f*self.norm.img)), g)
            f = primary_block(self.iradon(znorm.unnormalize(h))/self.norm.img, f)
            stages.append(torch.mean(f, dim=1, keepdim=True))

        if output_stages:
            return torch.mean(f*self.norm.img, dim=1, keepdim=True), stages

        return torch.mean(f*self.norm.img, dim=1, keepdim=True)


class PrimalDualNetworkSino(nn.Module):
    def __init__(self, in_size: int,
                 theta: Optional[Union[List[float], np.array, torch.tensor]],
                 n_primary: int, n_dual: int, n_iterations: int,
                 use_original_block: bool = True,
                 use_original_init: bool = True,
                 norm: Optional[SpaceNormalization] = None, 
                 fully_angles: Optional[Union[List[float], np.array, torch.tensor]] = None):
        super().__init__()
        self.primal_blocks = nn.ModuleList([
            (PrimalBlock if use_original_block else PrimalUnetBlock)(n_primary)
            for _ in range(n_iterations)
        ])
        self.dual_blocks = nn.ModuleList([
            DualBlock(n_dual) for _ in range(n_iterations)
        ])
        if fully_angles is None:
            fully_angles = list(range(180))
        self.radon = Radon(in_size, theta, circle=False, scikit=True)
        self.radon_end = Radon(in_size, fully_angles,
                               circle=False, scikit=True)
        self.iradon = IRadon(in_size, theta, circle=False,
                             use_filter=HannFilter(), scikit=True)
        self.in_size = in_size
        self.n_primary = n_primary
        self.n_dual = n_dual
        self.use_original_init = use_original_init
        self.norm = norm or SpaceNormalization(1.0, 1.0)

    def get_primal_dual_diff_weights(self):
        return {
            'primal': [f.diff_weight.mean().item() for f in self.primal_blocks],
            'dual': [f.diff_weight.mean().item() for f in self.dual_blocks]
        }

    def forward(self, sino: torch.tensor, sparse_reco: torch.tensor,
                output_stages: bool = False):
        # g, h: z-normed
        # f: per99-normed
        g = sino
        znorm = ZNorm(g)
        g = znorm.normalize(g)
        B, _, P, A = g.shape
        h = torch.zeros(B, self.n_dual, P, A, device=g.device)
        if self.use_original_init:
            f = torch.zeros(B, self.n_primary, self.in_size,
                            self.in_size, device=g.device)
        else:
            f = sparse_reco.repeat(1, self.n_primary, 1, 1)/self.norm.img

        stages = []
        for primary_block, dual_block in zip(self.primal_blocks, self.dual_blocks):
            h = dual_block(h, znorm.normalize(self.radon(f*self.norm.img)), g)
            f = primary_block(self.iradon(znorm.unnormalize(h))/self.norm.img, f)
            stages.append(torch.mean(f, dim=1, keepdim=True))

        if output_stages:
            return self.radon_end(torch.mean(f*self.norm.img, dim=1, keepdim=True)), stages

        full_reco = torch.mean(f*self.norm.img, dim=1, keepdim=True)
        full_sino = self.radon_end(full_reco)
        return full_sino, full_reco


def test_primal_dual_network():
    from torchsummary import summary
    theta = np.arange(180)[::8]
    net = PrimalDualNetwork(256, theta, 4, 128, 2).cuda()
    summary(net, (1, 363, 23))
    print(net(torch.zeros(2, 1, 363, 23).float().cuda()).shape)


if __name__ == '__main__':
    test_primal_dual_network()
