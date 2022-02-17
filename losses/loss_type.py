from enum import Enum, auto


class LossType(Enum):
    MSE = auto()
    L1 = auto()
    RESTORATION = auto()
