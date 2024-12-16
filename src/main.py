from datasets import DatasetManager
import models
from evaluator import ModelEvaluater


dataset_manager = DatasetManager()
x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = dataset_manager.get_dataset("MNIST")

model = models.StandardModel(x_train=x_train_mnist, y_train=y_train_mnist, learning_rate=0.01)
model.train(batch_size=128, epochs=1, verbose=1)

evaluator = ModelEvaluater(model, "MNIST", "FashionMNIST", threshold="pred_entropy")
evaluator.evaluate_data()

evaluator.evaluate_attack("l2pgd", dataset_type="OOD", epsilons=1.0)