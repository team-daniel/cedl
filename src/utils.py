from enum import Enum

class Datasets(Enum):
    MNIST = "MNIST"
    FashionMNIST = "FashionMNIST"
    KMNIST = "KMNIST"
    EMNIST = "EMNIST"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    DeepWeeds = "DeepWeeds"
    FLOWERS = "FLOWERS"
    SVHN = "SVHN"

class Thresholds(Enum):
    PRED_ENTROPY = "pred_entropy"
    DIFF_ENTROPY = "diff_entropy"
    TOTAL_ALPHA = "total_alpha"
    MUTUAL_INFO = "mutual_info"

class Attacks(Enum):
    L2PGD = "l2pgd"
    FGSM = "fgsm"