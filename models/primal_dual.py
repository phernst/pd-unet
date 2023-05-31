from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_radon import RadonFanbeam
from torch_radon.radon import ParallelBeam
from torch_radon.volumes import Volume2D

from utilities.transforms import ZNorm, filter_sinogram
from utilities.datasets import SpaceNormalization
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

    def _lr_padding(self, inp: torch.Tensor):
        return F.pad(
            inp, (self.kernel_size[1]//2, self.kernel_size[1]//2, 0, 0), mode='replicate')

    def _tb_padding(self, inp: torch.Tensor):
        return F.pad(
            inp, (0, 0, self.kernel_size[0]//2, self.kernel_size[0]//2), mode='replicate')

    def forward(self, x: torch.Tensor):
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

    def forward(self, h: torch.Tensor, f: torch.Tensor, g: torch.Tensor):
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

    def forward(self, h: torch.Tensor, f: torch.Tensor):
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
    def __init__(self, radon: Union[RadonFanbeam, ParallelBeam],
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
        self.radon = radon
        self.n_primary = n_primary
        self.n_dual = n_dual
        self.use_original_init = use_original_init
        self.norm = norm or SpaceNormalization(1.0, 1.0)

    def get_primal_dual_diff_weights(self):
        return {
            'primal': [f.diff_weight.mean().item() for f in self.primal_blocks],
            'dual': [f.diff_weight.mean().item() for f in self.dual_blocks]
        }

    def forward(self, sino: torch.Tensor, sparse_reco: torch.Tensor,
                output_stages: bool = False):
        # g, h: z-normed
        # f: per99-normed
        g = sino
        znorm = ZNorm(g)
        g = znorm.normalize(g)
        B, _, P, A = g.shape
        h = torch.zeros(B, self.n_dual, P, A, device=g.device)
        if self.use_original_init:
            f = torch.zeros(B, self.n_primary, sparse_reco.shape[-1],
                            sparse_reco.shape[-1], device=g.device)
        else:
            f = sparse_reco.repeat(1, self.n_primary, 1, 1)/self.norm.img

        stages = []
        for primary_block, dual_block in zip(self.primal_blocks, self.dual_blocks):
            h = dual_block(h, znorm.normalize(
                self.radon.forward(f*self.norm.img).transpose(-1, -2)), g)
            f = primary_block(self.radon.backprojection(filter_sinogram(
                znorm.unnormalize(h).transpose(-1, -2), 'hann'))/self.norm.img, f)
            stages.append(torch.mean(f, dim=1, keepdim=True))

        if output_stages:
            return torch.mean(f*self.norm.img, dim=1, keepdim=True), stages

        return torch.mean(f*self.norm.img, dim=1, keepdim=True)


class PrimalDualNetworkSino(nn.Module):
    def __init__(self, radon: Union[RadonFanbeam, ParallelBeam],
                 radon_end: Union[RadonFanbeam, ParallelBeam],
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
        self.radon = radon
        self.radon_end = radon_end
        self.n_primary = n_primary
        self.n_dual = n_dual
        self.use_original_init = use_original_init
        self.norm = norm or SpaceNormalization(1.0, 1.0)

    def get_primal_dual_diff_weights(self):
        return {
            'primal': [f.diff_weight.mean().item() for f in self.primal_blocks],
            'dual': [f.diff_weight.mean().item() for f in self.dual_blocks]
        }

    def forward(self, sino: torch.Tensor, sparse_reco: torch.Tensor,
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
            h = dual_block(h, znorm.normalize(
                self.radon.forward(f*self.norm.img).transpose(-1, -2)), g)
            f = primary_block(self.radon.backprojection(filter_sinogram(
                znorm.unnormalize(h).transpose(-1, -2), 'hann'))/self.norm.img, f)
            stages.append(torch.mean(f, dim=1, keepdim=True))

        if output_stages:
            return self.radon_end.forward(torch.mean(f*self.norm.img, dim=1, keepdim=True)).tranpose(-1, -2), stages

        full_reco = torch.mean(f*self.norm.img, dim=1, keepdim=True)
        full_sino = self.radon_end.forward(full_reco).transpose(-1, -2)
        return full_sino, full_reco


def test_primal_dual_network():
    from torchinfo import summary
    from utilities.fan_geometry import default_fan_geometry
    geom = default_fan_geometry()
    radon = RadonFanbeam(
        256, np.deg2rad(np.arange(360)[::16]),
        source_distance=geom.source_distance,
        det_distance=geom.det_distance,
        det_count=geom.det_count,
        det_spacing=geom.det_spacing,
    )
    net = PrimalDualNetwork(radon, 4, 128, 2).cuda()
    summary(net, (1, 1, 363, 23))
    print(net(torch.zeros(2, 1, 363, 23).float().cuda()).shape)


def test_single_timing_orig_vs_unet(num_projections: int) -> Tuple[float, float]:
    import time
    print(num_projections)
    with torch.no_grad():
        radon = ParallelBeam(
            363, np.deg2rad(np.arange(num_projections)),
            volume=Volume2D(256),
            # source_distance=geom.source_distance,
            # det_distance=geom.det_distance,
            # det_count=geom.det_count,
            # det_spacing=geom.det_spacing,
        )
        # net_orig = PrimalDualNetwork(
        #     radon, 5, 5, 10,
        net_orig = PrimalDualNetwork(
            # radon, 301, 301, 10,
            radon, 5, 5, 150,
            use_original_block=True,
            use_original_init=True).cuda()
        sparse_sino = torch.zeros(4, 1, 363, len(
            radon.angles), dtype=torch.float, device='cuda')
        under_reco = torch.zeros(
            4, 1, 256, 256, dtype=torch.float, device='cuda')
        # warmup
        for _ in range(5):
            net_orig(sparse_sino, under_reco)
        it = 10
        all_times = []
        for _ in range(it):
            starttime = time.time()
            __ = net_orig(sparse_sino, under_reco)
            all_times.append(time.time() - starttime)
        pd_orig_time = np.median(all_times)

        del net_orig
        net_unet = PrimalDualNetwork(
            radon, 4, 5, 2, use_original_block=False,
            use_original_init=False).cuda()
        # warmup
        for _ in range(5):
            net_unet(sparse_sino, under_reco)
        all_times = []
        for _ in range(it):
            starttime = time.time()
            __ = net_unet(sparse_sino, under_reco)
            all_times.append(time.time() - starttime)
        pd_unet_time = np.median(all_times)

    return pd_orig_time, pd_unet_time


def test_timing_orig_vs_unet():
    from matplotlib import pyplot as plt
    all_num_projections = list(range(2, 45, 10))
    all_times = list(map(test_single_timing_orig_vs_unet, all_num_projections))
    print(all_times)
    plt.plot(all_num_projections, [t[0] for t in all_times])
    plt.plot(all_num_projections, [t[1] for t in all_times])
    plt.grid()
    plt.legend(["PD Network", "PD UNet"],
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def show_parameters(sparsity: int):
    with torch.no_grad():
        from torchinfo import summary
        from utilities.fan_geometry import default_fan_geometry
        Net = PrimalDualNetwork
        geom = default_fan_geometry()
        radon = RadonFanbeam(
            256, np.deg2rad(np.arange(360)),
            source_distance=geom.source_distance,
            det_distance=geom.det_distance,
            det_count=geom.det_count,
            det_spacing=geom.det_spacing,
        )
        theta = np.arange(360)[::sparsity]
        radon = RadonFanbeam(
            256, np.deg2rad(theta),
            source_distance=geom.source_distance,
            det_distance=geom.det_distance,
            det_count=geom.det_count,
            det_spacing=geom.det_spacing,
        )

        img_shape = (1, 1, 256, 256)

        def num_parameters_pdunet():
            net = Net(radon, 4, 5, 2,
                      use_original_block=False,
                      use_original_init=False,
                      )
            sino_shape = net.radon.forward(
                torch.zeros(*img_shape, device='cuda')).transpose(-1, -2).shape
            summ = summary(net, (sino_shape, img_shape), verbose=0)
            return summ.total_params, summ.to_megabytes(summ.total_output_bytes)

        def num_parameters_pdorig():
            net = Net(radon, 5, 5, 10,
                      # net = Net(radon, 5, 5, 150,
                      # net = Net(radon, 301, 301, 10,
                      use_original_block=True,
                      use_original_init=True,
                      )
            sino_shape = net.radon.forward(
                torch.zeros(*img_shape, device='cuda')).transpose(-1, -2).shape
            summ = summary(net, (sino_shape, img_shape), verbose=0)
            return summ.total_params, summ.to_megabytes(summ.total_output_bytes)

        print(f'sparse {sparsity}, pdunet: {num_parameters_pdunet()}')
        print(f'sparse {sparsity}, pdorig: {num_parameters_pdorig()}')


def show_complexity(sparsity: int):
    with torch.no_grad():
        from thop import profile
        from thop import clever_format
        from utilities.fan_geometry import default_fan_geometry
        Net = PrimalDualNetwork
        geom = default_fan_geometry()
        radon = RadonFanbeam(
            256, np.deg2rad(np.arange(360)),
            source_distance=geom.source_distance,
            det_distance=geom.det_distance,
            det_count=geom.det_count,
            det_spacing=geom.det_spacing,
        )
        theta = np.arange(360)[::sparsity]
        radon = RadonFanbeam(
            256, np.deg2rad(theta),
            source_distance=geom.source_distance,
            det_distance=geom.det_distance,
            det_count=geom.det_count,
            det_spacing=geom.det_spacing,
        )

        img_shape = (1, 1, 256, 256)

        def num_parameters_pdunet():
            net = Net(radon, 4, 5, 2,
                      use_original_block=False,
                      use_original_init=False,
                      )
            sino_shape = net.radon.forward(
                torch.zeros(*img_shape, device='cuda')).transpose(-1, -2).shape
            macs, params = profile(
                net.cuda(), inputs=(torch.rand(sino_shape, device='cuda'),
                                    torch.rand(img_shape, device='cuda')))
            return clever_format([macs, params], "%.3f")

        def num_parameters_pdorig():
            # net = Net(radon, 5, 5, 10,
            # net = Net(radon, 5, 5, 150,
            net = Net(radon, 301, 301, 10,
                      use_original_block=True,
                      use_original_init=True,
                      )
            sino_shape = net.radon.forward(
                torch.zeros(*img_shape, device='cuda')).transpose(-1, -2).shape
            macs, params = profile(
                net.cuda(), inputs=(torch.rand(sino_shape, device='cuda'),
                                    torch.rand(img_shape, device='cuda')))
            return clever_format([macs, params], "%.3f")

        print(f'sparse {sparsity}, pdunet: {num_parameters_pdunet()}')
        print(f'sparse {sparsity}, pdorig: {num_parameters_pdorig()}')


if __name__ == '__main__':
    show_complexity(32)
    # test_timing_orig_vs_unet()
