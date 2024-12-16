from enum import Enum

class Datasets(Enum):
    MNIST = "MNIST"
    FashionMNIST = "FashionMNIST"

class Thresholds(Enum):
    PRED_ENTROPY = "pred_entropy"
    DIFF_ENTROPY = "diff_entropy"

class Attacks(Enum):
    L2PGD = "l2pgd"
    LinfPGD = "linfpgd"