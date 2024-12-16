import tensorflow as tf
import numpy as np
import sklearn

class StandardModel:
    def __init__(self, x_train, y_train, learning_rate=0.01):
        self.x_train = x_train
        self.y_train = y_train
        self.input_shape = x_train.shape[1:]
        self.learning_rate = learning_rate

        self.num_classes = y_train.shape[1]

        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=self.optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def _build_model(self):
        input = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)  # Softmax for probabilities
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose)
        return history
    
    def predict(self, inputs, verbose=0):
        return self.model.predict(inputs, verbose=verbose)

    def __call__(self, inputs):
        return self.model(inputs)
    
class EvidentialModel:
    def __init__(self, x_train, y_train, kl_weight=0.0001, learning_rate=0.01):
        self.x_train = x_train
        self.y_train = y_train
        self.input_shape = x_train.shape[1:]
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate

        self.num_classes = y_train.shape[1]

        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=self.optimizer,
                           loss=self._evidential_loss(),
                           metrics=['accuracy'])

    def _evidential_loss(self):
        def loss_fn(y_true, outputs):
            evidence = outputs
            alpha = evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            m = alpha / S

            err = tf.reduce_sum((y_true - m) ** 2, axis=1)
            var = tf.reduce_sum(m * (1 - m) / (S + 1), axis=1)
            loss = err + var

            kl_divergence = self.kl_weight * tf.reduce_sum((alpha - 1), axis=1)
            return tf.reduce_mean(loss + kl_divergence)
        return loss_fn

    def _build_model(self):
        input = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(self.num_classes, activation='softplus')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose)
        return history

    # get alphas
    def predict(self, inputs, verbose=0):
        evidence = self.model.predict(inputs, verbose=verbose)
        alpha = evidence + 1
        return alpha

    # get softmax like probs (needed for foolbox integration only)
    def __call__(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities