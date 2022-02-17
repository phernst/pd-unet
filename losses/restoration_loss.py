import torch
from torch.nn.modules.loss import _Loss
from torch.nn.functional import conv2d
import piq


class RestorationLoss(_Loss):
    def __init__(self, alpha: float, data_range: float, **kwargs):
        super().__init__()

        self.data_range = data_range
        self.msssim = piq.MultiScaleSSIMLoss(data_range=data_range, **kwargs)
        self.l1loss = torch.nn.L1Loss(reduction='none')
        self.gauss = piq.functional.gaussian_filter(
            self.msssim.kernel_size,
            self.msssim.kernel_sigma)
        self.alpha = alpha

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        msssim_metric = self.msssim(torch.clamp(
            prediction, min=0, max=self.data_range), target)

        l1map = self.l1loss(prediction, target)
        convl1map = conv2d(
            l1map,
            weight=self.gauss.repeat(prediction.size(1), 1, 1, 1).to(target),
            stride=1,
            padding=0,
            groups=prediction.size(1))
        convl1metric = torch.mean(convl1map)

        return self.alpha*msssim_metric + (1 - self.alpha)*convl1metric


def test_restoration_loss():
    loss = RestorationLoss()
    t1 = torch.zeros(1, 1, 256, 256, device='cuda')
    t2 = torch.ones(1, 1, 256, 256, device='cuda')
    print(loss(t1, t1))
    print(loss(t1, t2))


if __name__ == '__main__':
    test_restoration_loss()
