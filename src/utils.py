from enum import Enum

class Datasets(Enum):
    MNIST = "MNIST"
    FashionMNIST = "FashionMNIST"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    DeepWeeds = "DeepWeeds"
    CitrusLeaves = "CitrusLeaves"

class Thresholds(Enum):
    PRED_ENTROPY = "pred_entropy"
    DIFF_ENTROPY = "diff_entropy"
    TOTAL_ALPHA = "total_alpha"

class Attacks(Enum):
    L2PGD = "l2pgd"
    LinfPGD = "linfpgd"