from datasets import DatasetManager

dataset_manager = DatasetManager()
x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = dataset_manager.get_dataset("MNIST")

print(x_train_mnist.shape)