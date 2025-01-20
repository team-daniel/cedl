from datasets import DatasetManager
from utils import Datasets, Thresholds, Attacks
import models
from evaluator import ModelEvaluator

import numpy as np
import time

id_accuracy, id_coverage, id_delta = [], [], []
ood_coverage, ood_delta = [], []
adv_coverage, adv_delta, adv_perturbation = [], [], []
train_times, eval_times = [], []

runs = 1

for i in range(runs):
    print(f"Run: {i}")

    dataset_manager = DatasetManager()
    x_train_mnist, y_train_mnist, _, _ = dataset_manager.get_dataset(Datasets.MNIST)
    x_train_fashion_mnist, y_train_fashion_mnist, _, _ = dataset_manager.get_dataset(Datasets.FashionMNIST)

    model = models.InformationEvidentialModel(x_train=x_train_mnist, y_train=y_train_mnist, learning_rate=0.01)

    start_train_time = time.time()
    model.train(batch_size=128, epochs=1, verbose=1)
    end_train_time = time.time()
    train_times.append(end_train_time - start_train_time)

    evaluator = ModelEvaluator(model, Datasets.MNIST, Datasets.FashionMNIST, threshold=Thresholds.DIFF_ENTROPY)

    start_eval_time = time.time()
    results = evaluator.evaluate_data()
    end_eval_time = time.time()
    eval_times.append(end_eval_time - start_eval_time)

    id_accuracy.append(results["ID"]["accuracy"])
    id_coverage.append(results["ID"]["coverage"])
    id_delta.append(results["ID"]["mean_evidence_delta"])
    ood_coverage.append(results["OOD"]["coverage"])
    ood_delta.append(results["OOD"]["mean_evidence_delta"])

    results = evaluator.evaluate_attack(Attacks.L2PGD, dataset_type="OOD", epsilons=1.0)

    adv_coverage.append(results["ADV"]["coverage"])
    adv_delta.append(results["ADV"]["mean_evidence_delta"])
    adv_perturbation.append(results["avg_perturbation"])

print("========================")
print(f"Mean Accuracy on ID data: {np.mean(id_accuracy) * 100:.4f} +/- {np.std(id_accuracy) * 100:.4f}")
print(f"Mean Coverage on ID data: {np.mean(id_coverage) * 100:.4f} +/- {np.std(id_coverage) * 100:.4f}")
print(f"Mean Delta on ID data: {np.mean(id_delta):.4f} +/- {np.std(id_delta):.4f}")
print("========================")
print(f"Mean Coverage on OOD data: {np.mean(ood_coverage) * 100:.4f} +/- {np.std(ood_coverage) * 100:.4f}")
print(f"Mean Delta on OOD data: {np.mean(ood_delta):.4f} +/- {np.std(ood_delta):.4f}")
print("========================")
print(f"Mean Coverage on ADV data: {np.mean(adv_coverage) * 100:.4f} +/- {np.std(adv_coverage) * 100:.4f}")
print(f"Mean Delta on ADV data: {np.mean(adv_delta):.4f} +/- {np.std(adv_delta):.4f}")
print(f"Mean Perturbation on ADV data: {np.mean(adv_perturbation):.4f} +/- {np.std(adv_perturbation):.4f}")
print("========================")
print(f"Mean Training Time: {np.mean(train_times):.4f} seconds +/- {np.std(train_times):.4f}")
print(f"Mean Evaluation Time: {np.mean(eval_times):.4f} seconds +/- {np.std(eval_times):.4f}")
print("========================")