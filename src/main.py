from datasets import DatasetManager
from utils import Datasets, Thresholds, Attacks
import models
from evaluator import ModelEvaluator


dataset_manager = DatasetManager()
x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = dataset_manager.get_dataset(Datasets.MNIST)

model = models.EvidentialModel(x_train=x_train_mnist, y_train=y_train_mnist, learning_rate=0.01)
model.train(batch_size=128, epochs=1, verbose=1)

evaluator = ModelEvaluator(model, Datasets.MNIST, Datasets.FashionMNIST, threshold=Thresholds.DIFF_ENTROPY)
evaluator.evaluate_data()

evaluator.evaluate_attack(Attacks.L2PGD, dataset_type="OOD", epsilons=1.0)