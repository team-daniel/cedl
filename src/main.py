from dataset import DatasetManager
from utils import Datasets, Thresholds, Attacks
import models
from evaluator import ClassificationEvaluator

import numpy as np
import time
import pickle
import tensorflow.keras.backend as K
import gc

id_accuracy, id_coverage, id_delta, id_below_delta, id_above_delta = [], [], [], [], []
ood_coverage, ood_delta, ood_below_delta, ood_above_delta = [], [], [], []
adv_coverage, adv_delta, adv_below_delta, adv_above_delta, adv_perturbation = [], [], [], [], []
train_times, eval_times = [], []

runs = 10

# Classification loop
for i in range(runs):
    print(f"Run: {i}")

    dataset_manager = DatasetManager()
    x_train_id, y_train_id, _, _, _, _ = dataset_manager.get_dataset(Datasets.CIFAR10)
    x_train_ood, y_train_ood, _, _, _, _ = dataset_manager.get_dataset(Datasets.CIFAR100)

    print(f"Training model...")
    model = models.PosteriorModel(x_train=x_train_id, y_train=y_train_id, learning_rate=0.001)

    start_train_time = time.time()
    model.train(batch_size=64, epochs=250, verbose=0)
    end_train_time = time.time()
    train_times.append(end_train_time - start_train_time)

    evaluator = ClassificationEvaluator(model, Datasets.CIFAR10, Datasets.CIFAR100, threshold=Thresholds.DIFF_ENTROPY)

    print(f"Evaluation model...")
    start_eval_time = time.time()
    results = evaluator.evaluate_data()
    end_eval_time = time.time()
    eval_times.append(end_eval_time - start_eval_time)

    id_accuracy.append(results["ID"]["accuracy"])
    id_coverage.append(results["ID"]["coverage"])
    id_delta.append(results["ID"]["mean_evidence_delta"])
    id_below_delta.append(results["ID"]["mean_evidence_below_delta"])
    id_above_delta.append(results["ID"]["mean_evidence_above_delta"])
    ood_coverage.append(results["OOD"]["coverage"])
    ood_delta.append(results["OOD"]["mean_evidence_delta"])
    ood_below_delta.append(results["OOD"]["mean_evidence_below_delta"])
    ood_above_delta.append(results["OOD"]["mean_evidence_above_delta"])

    print(f"Evaluating attack...")
    results = evaluator.evaluate_attack(Attacks.L2PGD, dataset_type="OOD", epsilons=[0.1])

    adv_coverage.append(results["ADV"]["coverage"])
    adv_delta.append(results["ADV"]["mean_evidence_delta"])
    adv_below_delta.append(results["ADV"]["mean_evidence_below_delta"])
    adv_above_delta.append(results["ADV"]["mean_evidence_above_delta"])
    adv_perturbation.append(results["avg_perturbation"])

    K.clear_session()
    gc.collect()
    del model
    del evaluator

print("========================")
print(f"Mean Accuracy on ID data: {np.mean(id_accuracy) * 100:.4f} +/- {np.std(id_accuracy) * 100:.4f}")
print(f"Mean Coverage on ID data: {np.mean(id_coverage) * 100:.4f} +/- {np.std(id_coverage) * 100:.4f}")
print(f"Mean Delta on ID data: {np.mean(id_delta):.4f} +/- {np.std(id_delta):.4f}")
print(f"Mean Delta below threshold on ID data: {np.mean(id_below_delta):.4f} +/- {np.std(id_below_delta):.4f}")
print(f"Mean Delta above threshold on ID data: {np.mean(id_above_delta):.4f} +/- {np.std(id_above_delta):.4f}")
print("========================")
print(f"Mean Coverage on OOD data: {np.mean(ood_coverage) * 100:.4f} +/- {np.std(ood_coverage) * 100:.4f}")
print(f"Mean Delta on OOD data: {np.mean(ood_delta):.4f} +/- {np.std(ood_delta):.4f}")
print(f"Mean Delta below threshold on OOD data: {np.mean(ood_below_delta):.4f} +/- {np.std(ood_below_delta):.4f}")
print(f"Mean Delta above threshold on OOD data: {np.mean(ood_above_delta):.4f} +/- {np.std(ood_above_delta):.4f}")
print("========================")
print(f"Mean Coverage on ADV data: {np.mean(adv_coverage) * 100:.4f} +/- {np.std(adv_coverage) * 100:.4f}")
print(f"Mean Delta on ADV data: {np.mean(adv_delta):.4f} +/- {np.std(adv_delta):.4f}")
print(f"Mean Delta below threshold on ADV data: {np.mean(adv_below_delta):.4f} +/- {np.std(adv_below_delta):.4f}")
print(f"Mean Delta above threshold on ADV data: {np.mean(adv_above_delta):.4f} +/- {np.std(adv_above_delta):.4f}")
print(f"Mean Perturbation on ADV data: {np.mean(adv_perturbation):.4f} +/- {np.std(adv_perturbation):.4f}")
print("========================")
print(f"Mean Training Time: {np.mean(train_times):.4f} seconds +/- {np.std(train_times):.4f}")
print(f"Mean Evaluation Time: {np.mean(eval_times):.4f} seconds +/- {np.std(eval_times):.4f}")
print("========================")

print("Saving results...")
results_dict = {
    "id_accuracy": id_accuracy,
    "id_coverage": id_coverage,
    "id_delta": id_delta,
    "id_below_delta": id_below_delta,
    "id_above_delta": id_above_delta,
    "ood_coverage": ood_coverage,
    "ood_delta": ood_delta,
    "ood_below_delta": ood_below_delta,
    "ood_above_delta": ood_above_delta,
    "adv_coverage": adv_coverage,
    "adv_delta": adv_delta,
    "adv_below_delta": adv_below_delta,
    "adv_above_delta": adv_above_delta,
    "adv_perturbation": adv_perturbation,
    "train_times": train_times,
    "eval_times": eval_times
}

with open("Results/cifar10_cifar100_postnet", "wb") as f:
    pickle.dump(results_dict, f)
