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
- - Evidential++ Network
- - - Similar to the Conflicting Evidential Network but without the conflicting evidence reduction, only the metamorphic transformation.
- - Conflicting Evidential Network

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
- - MNIST
- - FashionMNIST
- - CIFAR10
- - CIFAR100
- - DeepWeeds
- - CitrusLeaves

## Required Packages

If you would like to run this code, please install these versions of the following packages:

```python
numpy==2.2.4
eagerpy==0.30.0
foolbox==3.3.4
matplotlib==3.10.1
tensorflow==2.19.0
scipy==1.15.2
tqdm==4.67.1
scikit-learn==1.2.1
opencv-python==4.11.0.86
tensorflow-datasets==4.9.2
datasets==3.1.0
bs4==0.0.2
```
