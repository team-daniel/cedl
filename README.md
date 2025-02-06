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

## Attacks

Currently support:

- L2PGD
- LinfPGD

## Datasets

Currently support:

- Classification Datasets
- - MNIST
- - FashionMNIST
