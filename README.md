# C-EDL

text here

## Models

Currently support:

- Classification Models
- - Deep Neural Network
- - MCDropout Network
- - - Uses Adaptive MC Dropout (Bethell, et al. 2024) to ensures enough fwd passes.
- - Posterior Network
- - Evidential Network
- - Fisher-Information Evidential Network
- - Smoothed Evidential Network
- - Hyper-opinion Evidential Network
- - Relaxed Evidential Network
- - Density Aware Evidential Network
- - Evidential++ Network (Meta)
- - - Similar to the Conflicting Evidential Network but without the conflicting evidence reduction, only the metamorphic transformation.
- - Evidential++ Network (MC)
- - - Similar to the Conflicting Evidential Network but without the conflicting evidence reduction, only the MC Dropout.
- - Conflicting Evidential Network (Meta)
- - Conflicting Evidential Network (MC)

## Threshold Metrics

Currently support:

- Predictive Entropy
- Differential Entropy
- Total Alpha

## Attacks

Currently support:

- L2PGD
- LinfPGD

## Datasets

Currently support:

- Classification Datasets
- (28 x 28 images)
- - MNIST
- - FashionMNIST
- - KMNIST
- - EMNIST
- (32 x 32 images)
- - CIFAR10
- - CIFAR100
- - DeepWeeds
- - CitrusLeaves

## Required Packages

If you would like to run this code, please install these versions of the following packages:

```python
numpy==1.26.4
eagerpy==0.30.0
foolbox==3.3.4
matplotlib==3.10.1
tensorflow==2.15.0
scipy==1.15.2
tqdm==4.67.1
scikit-learn==1.2.1
opencv-python==4.11.0.86
tensorflow-datasets==4.9.8
datasets==3.1.0
bs4==0.0.2
```
