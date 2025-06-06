from .dataset import DatasetManager
from .utils import Datasets, Thresholds, Attacks
from . import metrics
from . import models

import eagerpy as ep
import foolbox
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gc

class ClassificationEvaluator:
    def __init__(self, model, id_dataset_name: Datasets, ood_dataset_name: Datasets, threshold: Thresholds):
        self.dataset_manager = DatasetManager()
        self.id_dataset_name = id_dataset_name
        self.ood_dataset_name = ood_dataset_name

        _, _, self.id_x_test, self.id_y_test, self.id_x_val, self.id_y_val = self.dataset_manager.get_dataset(self.id_dataset_name)
        _, _, self.ood_x_test, self.ood_y_test, self.ood_x_val, self.ood_y_val = self.dataset_manager.get_dataset(self.ood_dataset_name)
        self._check_dataset_shapes()

        self.model = model
        self.fmodel = foolbox.models.TensorFlowModel(self.model, bounds=(0, 1))
        self.threshold = threshold
        self.optimal_threshold = None

    def _check_dataset_shapes(self):
        if self.id_x_test.shape[1:] != self.ood_x_test.shape[1:]:
            raise ValueError(
                f"Mismatch in input shapes between ID and OOD datasets:\n"
                f"ID dataset input shape: {self.id_x_test.shape[1:]}\n"
                f"OOD dataset input shape: {self.ood_x_test.shape[1:]}"
            )

    # get the ID-OOD threshold
    def get_threshold(self):
        if self.threshold == Thresholds.DIFF_ENTROPY and not isinstance(self.model, (models.EvidentialModel, models.SmoothedEvidentialModel, models.InformationEvidentialModel, models.HyperEvidentialModel, models.RelaxedEvidentialModel, models.DensityAwareEvidentialModel, models.EvidentialPlusMetaModel, models.EvidentialPlusMcModel, models.ConflictingEvidentialMetaModel, models.ConflictingEvidentialMcModel, models.PosteriorModel)):
                raise RuntimeError("Differential Entropy threshold is only allowed for Evidential-based models.")
        
        if self.threshold == Thresholds.TOTAL_ALPHA and not isinstance(self.model, (models.EvidentialModel, models.SmoothedEvidentialModel, models.InformationEvidentialModel, models.HyperEvidentialModel, models.RelaxedEvidentialModel, models.DensityAwareEvidentialModel, models.EvidentialPlusMetaModel, models.EvidentialPlusMcModel, models.ConflictingEvidentialMetaModel, models.ConflictingEvidentialMcModel, models.PosteriorModel)):
                raise RuntimeError("Total Alpha threshold is only allowed for Evidential-based models.")
        
        if self.threshold == Thresholds.PRED_ENTROPY:
            if isinstance(self.model, (models.EvidentialPlusMetaModel, models.EvidentialPlusMcModel, models.ConflictingEvidentialMetaModel, models.ConflictingEvidentialMcModel)): 
                alpha_id = self.model.predict_probs(self.id_x_val, for_threshold=True)
                alpha_ood = self.model.predict_probs(self.ood_x_val, for_threshold=True)
            else:
                alpha_id = self.model.predict_probs(self.id_x_val)
                alpha_ood = self.model.predict_probs(self.ood_x_val)
        else:
            if isinstance(self.model, (models.EvidentialPlusMetaModel, models.EvidentialPlusMcModel, models.ConflictingEvidentialMetaModel, models.ConflictingEvidentialMcModel)): 
                alpha_id = self.model.predict(self.id_x_val, for_threshold=True)
                alpha_ood = self.model.predict(self.ood_x_val, for_threshold=True)
            else:
                alpha_id = self.model.predict(self.id_x_val, verbose=0)
                alpha_ood = self.model.predict(self.ood_x_val, verbose=0)
        _, _, _, _, optimal_threshold = metrics.get_optimal_threshold(alpha_id, alpha_ood, metric=self.threshold)

        return optimal_threshold

    # evaluate the Id and OOD data
    def evaluate_data(self):
        if self.threshold == Thresholds.PRED_ENTROPY:
            id_predictions = self.model.predict_probs(self.id_x_test)
            ood_predictions = self.model.predict_probs(self.ood_x_test)
        else:
            id_predictions = self.model.predict(self.id_x_test, verbose=0)
            ood_predictions = self.model.predict(self.ood_x_test, verbose=0)
        
        self.optimal_threshold = self.get_threshold()
        if self.threshold == Thresholds.PRED_ENTROPY:
            self.optimal_threshold = -self.optimal_threshold
          
        id_accuracy, id_coverage, id_mean_evidence_delta, id_mean_evidence_below_delta, id_mean_evidence_above_delta = self.get_results(id_predictions, self.id_y_test, "ID")
        _, ood_coverage, ood_mean_evidence_delta, ood_mean_evidence_below_delta, ood_mean_evidence_above_delta = self.get_results(ood_predictions, self.ood_y_test, "OOD")

        results = {
            "ID": {
                "accuracy": id_accuracy,
                "coverage": id_coverage,
                "mean_evidence_delta": id_mean_evidence_delta,
                "mean_evidence_below_delta": id_mean_evidence_below_delta,
                "mean_evidence_above_delta": id_mean_evidence_above_delta,
            },
            "OOD": {
                "coverage": ood_coverage,
                "mean_evidence_delta": ood_mean_evidence_delta,
                "mean_evidence_below_delta": ood_mean_evidence_below_delta,
                "mean_evidence_above_delta": ood_mean_evidence_above_delta,
            },
            "optimal_threshold": self.optimal_threshold
        }
        return results

    # evaluate either dataset when attacked using foolbox
    def evaluate_attack(self, attack, dataset_type="ID", epsilons=1.5):
        if dataset_type == "ID":
            images = ep.astensors(tf.convert_to_tensor(self.id_x_test))[0]
            labels = ep.astensors(tf.convert_to_tensor(np.argmax(self.id_y_test, axis=1)))[0]
            true_labels = self.id_y_test
        elif dataset_type == "OOD":
            images = ep.astensors(tf.convert_to_tensor(self.ood_x_test))[0]
            if isinstance(self.model, (models.EvidentialPlusMetaModel, models.EvidentialPlusMcModel, models.ConflictingEvidentialMetaModel, models.ConflictingEvidentialMcModel)):
                pseudo_labels = tf.argmax(self.model.predict(self.ood_x_test, for_threshold=True), axis=1)
            else:
                pseudo_labels = tf.argmax(self.model.predict(self.ood_x_test), axis=1)
            labels = ep.astensor(tf.convert_to_tensor(pseudo_labels))
            true_labels = self.ood_y_test
        else:
            raise ValueError("dataset_type must be either 'ID' or 'OOD'.")

        if attack == Attacks.L2PGD:
            attack = foolbox.attacks.L2PGD()
        elif attack == Attacks.FGSM:
            attack = foolbox.attacks.FGSM()

        else:
            raise ValueError("Unsupported attack.")

        criterion = foolbox.criteria.Misclassification(labels)
        _, adversarial_images, _ = attack(self.fmodel, images, criterion, epsilons=epsilons)
        adversarial_images = adversarial_images[0].numpy()

        if self.threshold == Thresholds.PRED_ENTROPY:
            adv_predictions = self.model.predict_probs(adversarial_images)
        else:
            adv_predictions = self.model.predict(adversarial_images, verbose=0)

        if not self.optimal_threshold: self.evaluate_data()
        _, adv_coverage, adv_mean_evidence_delta, adv_mean_evidence_below_delta, adv_mean_evidence_above_delta = self.get_results(adv_predictions, true_labels, f"Adversarial {dataset_type}")

        perturbations = adversarial_images - images.numpy()
        perturbation_norms = tf.norm(tf.reshape(perturbations, (perturbations.shape[0], -1)), axis=1)
        avg_perturbation_norm = tf.reduce_mean(perturbation_norms).numpy()

        results = {
            "ADV": {
                "coverage": adv_coverage,
                "mean_evidence_delta": adv_mean_evidence_delta,
                "mean_evidence_below_delta": adv_mean_evidence_below_delta,
                "mean_evidence_above_delta": adv_mean_evidence_above_delta
            },
            "avg_perturbation": avg_perturbation_norm
        }

        del images, labels, adversarial_images, adv_predictions, criterion
        tf.keras.backend.clear_session()
        gc.collect()

        return results

    # get results based off predicitons
    def get_results(self, predictions, true_labels, dataset_type):
        if self.threshold == Thresholds.DIFF_ENTROPY:
            scores = metrics.diff_entropy(predictions)
            decision = scores >= self.optimal_threshold
        elif self.threshold == Thresholds.PRED_ENTROPY:
            scores = -metrics.pred_entropy(predictions)
            decision = scores <= self.optimal_threshold 
        elif self.threshold == Thresholds.TOTAL_ALPHA:
            scores = metrics.total_alpha(predictions)
            decision = scores >= self.optimal_threshold
        elif self.threshold == Thresholds.MUTUAL_INFO:
            scores = metrics.mutual_info(predictions)
            decision = scores >= self.optimal_threshold

        indices = tf.convert_to_tensor(np.where(decision)[0], dtype=tf.int32)

        if len(indices) == 0:
            print(f"No predictions made for {dataset_type}. Coverage: 0.00%")
            mean_evidence_delta = np.mean(scores - self.optimal_threshold)
            below_threshold_mask = scores < self.optimal_threshold
            remaining_scores = scores[below_threshold_mask] - self.optimal_threshold
            mean_evidence_below_delta = np.mean(remaining_scores) if remaining_scores.size > 0 else 0.0

            above_threshold_mask = scores > self.optimal_threshold
            remaining_scores = scores[above_threshold_mask] - self.optimal_threshold
            mean_evidence_above_delta = np.mean(remaining_scores) if remaining_scores.size > 0 else 0.0
            return 0.0, 0.0, mean_evidence_delta, mean_evidence_below_delta, mean_evidence_above_delta
        
        predictions_made = tf.gather(predictions, indices)
        labels_made = tf.gather(true_labels, indices)

        if predictions_made.shape.ndims > 1:
            predictions_made = tf.argmax(predictions_made, axis=1)
            labels_made = tf.argmax(labels_made, axis=1)

        accuracy = tf.reduce_mean(tf.cast(predictions_made == labels_made, tf.float32)).numpy()
        coverage = np.sum(decision) / len(scores)

        delta = scores - self.optimal_threshold
        mean_evidence_all_delta = np.mean(delta)

        below_threshold_mask = scores < self.optimal_threshold
        remaining_scores = scores[below_threshold_mask] - self.optimal_threshold
        mean_evidence_below_delta = np.mean(remaining_scores) if remaining_scores.size > 0 else 0.0

        above_threshold_mask = scores > self.optimal_threshold
        remaining_scores = scores[above_threshold_mask] - self.optimal_threshold
        mean_evidence_above_delta = np.mean(remaining_scores) if remaining_scores.size > 0 else 0.0

        return accuracy, coverage, mean_evidence_all_delta, mean_evidence_below_delta, mean_evidence_above_delta 
