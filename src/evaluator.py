from datasets import DatasetManager
import models
import metrics

import foolbox
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class ModelEvaluater:
    def __init__(self, model, id_dataset_name, ood_dataset_name, threshold="diff_entropy"):
        self.dataset_manager = DatasetManager()
        self.id_dataset_name = id_dataset_name
        self.ood_dataset_name = ood_dataset_name

        self.id_x_train, self.id_y_train, self.id_x_test, self.id_y_test = self.dataset_manager.get_dataset(self.id_dataset_name)
        self.ood_x_train, self.ood_y_train, self.ood_x_test, self.ood_y_test = self.dataset_manager.get_dataset(self.ood_dataset_name)
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
        if self.threshold == "diff_entropy" and not isinstance(self.model, (models.EvidentialModel, models.EvidentialPlusModel, models.ConflictingEvidentialModel)):
                raise RuntimeError("Differential Entropy threshold is only allowed for Evidential-based models.")
        
        if self.threshold == "pred_entropy":
            if isinstance(self.model, (models.EvidentialPlusModel, models.ConflictingEvidentialModel)): 
                alpha_id = self.model.predict_probs(self.id_x_test, for_threshold=True)
                alpha_ood = self.model.predict_probs(self.ood_x_test, for_threshold=True)
            else:
                alpha_id = self.model.predict_probs(self.id_x_test)
                alpha_ood = self.model.predict_probs(self.ood_x_test)
        else:
            if isinstance(self.model, (models.EvidentialPlusModel, models.ConflictingEvidentialModel)): 
                alpha_id = self.model.predict(self.id_x_test, for_threshold=True)
                alpha_ood = self.model.predict(self.ood_x_test, for_threshold=True)
            else:
                alpha_id = self.model.predict(self.id_x_test, verbose=0)
                alpha_ood = self.model.predict(self.ood_x_test, verbose=0)
        _, _, _, _, optimal_threshold = metrics.get_optimal_threshold(alpha_id, alpha_ood, metric=self.threshold)

        return optimal_threshold

    # evaluate the Id and OOD data
    def evaluate_data(self):
        if self.threshold == "pred_entropy":
            id_predictions = self.model.predict_probs(self.id_x_test)
            ood_predictions = self.model.predict_probs(self.ood_x_test)
        else:
            id_predictions = self.model.predict(self.id_x_test, verbose=0)
            ood_predictions = self.model.predict(self.ood_x_test, verbose=0)
        
        self.optimal_threshold = self.get_threshold()
        if self.threshold == "pred_entropy":
            self.optimal_threshold = -self.optimal_threshold
        print(f"Optimal Threshold: {self.optimal_threshold:.4f}") 
          
        self.get_results(id_predictions, self.id_y_test, "ID")
        self.get_results(ood_predictions, self.ood_y_test, "OOD")

    # evaluate either dataset when attacked using foolbox
    def evaluate_attack(self, attack, dataset_type="ID", epsilons=1.5):
        if dataset_type == "ID":
            images = tf.convert_to_tensor(self.id_x_test)
            labels = tf.convert_to_tensor(np.argmax(self.id_y_test, axis=1))
            true_labels = self.id_y_test
        elif dataset_type == "OOD":
            images = tf.convert_to_tensor(self.ood_x_test)
            labels = tf.convert_to_tensor(np.argmax(self.ood_y_test, axis=1))
            true_labels = self.ood_y_test
        else:
            raise ValueError("dataset_type must be either 'ID' or 'OOD'.")

        if attack == "l2pgd":
            attack = foolbox.attacks.L2PGD()
        elif attack == "linfpgd":
            attack = foolbox.attacks.LinfPGD()
        else:
            raise ValueError("Unsupported attack.")

        criterion = foolbox.criteria.Misclassification(labels)
        _, adversarial_images, _ = attack(self.fmodel, images, criterion, epsilons=epsilons)

        if self.threshold == "pred_entropy":
            adv_predictions = self.model.predict_probs(adversarial_images)
        else:
            adv_predictions = self.model.predict(adversarial_images, verbose=0)

        if not self.optimal_threshold: self.evaluate_data()
        self.get_results(adv_predictions, true_labels, f"Adversarial {dataset_type}")

        perturbations = adversarial_images - images
        perturbation_norms = tf.norm(tf.reshape(perturbations, (perturbations.shape[0], -1)), axis=1)
        avg_perturbation_norm = tf.reduce_mean(perturbation_norms).numpy()

        print(f"Average L2 Norm of Perturbation: {avg_perturbation_norm:.4f}")

    # get results based off predicitons
    def get_results(self, predictions, true_labels, dataset_type):
        if self.threshold == "diff_entropy":
            scores = metrics.diff_entropy(predictions)
            decision = scores >= self.optimal_threshold
        elif self.threshold == "pred_entropy":
            scores = -metrics.pred_entropy(predictions)
            decision = scores <= self.optimal_threshold 

        indices = tf.convert_to_tensor(np.where(decision)[0], dtype=tf.int32)

        if len(indices) == 0:
                print(f"No predictions made for {dataset_type}. Coverage: 0.00%")
                return float('nan'), 0.0
        
        predictions_made = tf.gather(predictions, indices)
        labels_made = tf.gather(true_labels, indices)

        if predictions_made.shape.ndims > 1:
            predictions_made = tf.argmax(predictions_made, axis=1)
            labels_made = tf.argmax(labels_made, axis=1)

        accuracy = tf.reduce_mean(tf.cast(predictions_made == labels_made, tf.float32)).numpy()
        coverage = np.sum(decision) / len(scores)
        delta = scores - self.optimal_threshold
        mean_evidence_delta = np.mean(delta)

        print(f"Mean Delta on {dataset_type} data: {mean_evidence_delta:.4f}")
        print(f"Accuracy on {dataset_type} data: {accuracy * 100:.2f}%")
        print(f"Coverage on {dataset_type} data: {coverage * 100:.2f}%")
