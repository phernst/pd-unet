import torch
from torch.nn.modules.loss import _Loss


class RMSELoss(_Loss):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return self.mse(prediction, target).sqrt()


def test_rmse_loss():
    loss = RMSELoss()
    t1 = torch.zeros(1, 1, 256, 256, device='cuda')
    t2 = torch.ones(1, 1, 256, 256, device='cuda')
    print(loss(t1, t1))
    print(loss(t1, t2))


if __name__ == '__main__':
    test_rmse_loss()
