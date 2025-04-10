import scipy.ndimage
import tensorflow as tf
import numpy as np
import random
import scipy
from tqdm import tqdm
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

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose
        )
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose,
                                 callbacks=[lr_scheduler])
        return history

    def predict(self, inputs, verbose=0):
        return self.model.predict(inputs, verbose=verbose)

    def predict_probs(self, inputs):
        probabilities = self.model(inputs)
        return probabilities

    def __call__(self, inputs):
        return self.model(inputs)
    
class MCDropoutModel:
    def __init__(self, x_train, y_train, learning_rate=0.01):
        self.x_train = x_train
        self.y_train = y_train
        self.input_shape = x_train.shape[1:]
        self.learning_rate = learning_rate

        self.num_classes = y_train.shape[1]

        self.delta = 1e-4
        self.patience = 10
        self.num_mc_samples = 1000

        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=self.optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def _build_model(self):
        input = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose
        )
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose,
                                 callbacks=[lr_scheduler])
        return history

    def adaptive_mc_predict(self, inputs):
        num_samples = len(inputs)
        mc_preds_list = [[] for _ in range(num_samples)]
        prev_variances = [None] * num_samples
        patience_counts = [0] * num_samples
        active_indices = list(range(num_samples))
        for i in tqdm(range(self.num_mc_samples), desc="Processing inputs"):
            if not active_indices:
                break
            active_inputs = inputs[active_indices]
            preds = self.model(active_inputs, training=True).numpy()
            to_remove = []
            for j, idx in enumerate(active_indices):
                mc_preds_list[idx].append(preds[j])
                if len(mc_preds_list[idx]) > 1:
                    current_variance = np.std(mc_preds_list[idx], axis=0)
                    if prev_variances[idx] is not None:
                        var_diff = np.abs(current_variance - prev_variances[idx])
                        if np.all(var_diff <= self.delta):
                            patience_counts[idx] += 1
                        else:
                            patience_counts[idx] = 0

                        if patience_counts[idx] >= self.patience:
                            to_remove.append(idx)
                    prev_variances[idx] = current_variance
            active_indices = [idx for idx in active_indices if idx not in to_remove]
        mean_preds = [np.mean(mc_preds, axis=0) for mc_preds in mc_preds_list]
        mean_fwd_passes = [len(mc_preds) for mc_preds in mc_preds_list]
        print("Mean number of forward passes per input:", np.mean(mean_fwd_passes))
        return np.array(mean_preds)

    def predict(self, inputs, verbose=0):
        return self.adaptive_mc_predict(inputs)

    def predict_probs(self, inputs):
        probabilities = self.adaptive_mc_predict(inputs)
        return probabilities

    def __call__(self, inputs):
        return self.model(inputs)

class PosteriorModel:
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
                           loss=self._posterior_loss(),
                           metrics=['accuracy'])

    def _posterior_loss(self):
        def loss_fn(y_true, outputs):
            alpha = outputs + 1.0
            alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
            pred = alpha / alpha_sum

            ce = tf.keras.losses.categorical_crossentropy(y_true, pred)

            k = tf.cast(tf.shape(alpha)[1], tf.float32)
            sum_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)

            term1 = tf.math.lgamma(sum_alpha) - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
            term2 = tf.reduce_sum(tf.math.lgamma(tf.ones_like(alpha)), axis=1, keepdims=True) - tf.math.lgamma(k)
            term3 = tf.reduce_sum((alpha - 1.0) * (tf.math.digamma(alpha) - tf.math.digamma(sum_alpha)), axis=1, keepdims=True)
            kl = term1 + term2 + term3
            kl = tf.reduce_sum(kl) / tf.cast(tf.shape(y_true)[0], tf.float32)

            return ce + self.kl_weight * kl
        return loss_fn

    def _build_model(self):
        input = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(self.num_classes, activation='softplus')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        class FisherWeightCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                self.parent = parent

            def on_epoch_begin(self, epoch, logs=None):
                self.parent.kl_weight = min(1.0, epoch / 10.0)

        fisher_callback = FisherWeightCallback(self)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose
        )
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose,
                                 callbacks=[fisher_callback, lr_scheduler])
        return history

    # get alphas
    def predict(self, inputs, verbose=0):
        evidence = self.model.predict(inputs, verbose=verbose)
        alpha = evidence + 1
        return alpha

    def predict_probs(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities

    # get softmax like probs (needed for foolbox integration only)
    def __call__(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities
     
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

    def dirichlet_kl_divergence(self, alpha):
        alpha_shape = tf.shape(alpha)
        K = tf.cast(alpha_shape[1], alpha.dtype)
        alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
        kl = tf.math.lgamma(alpha_sum) - tf.math.lgamma(K)
        kl -= tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
        kl += tf.reduce_sum(
            (alpha - 1.0) * (tf.math.digamma(alpha) - tf.math.digamma(alpha_sum)),
            axis=1,
            keepdims=True
        )
        return kl

    def _evidential_loss(self):
        def loss_fn(y_true, outputs):
            evidence = outputs
            alpha = evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            m = alpha / S

            err = tf.reduce_sum((y_true - m) ** 2, axis=1)
            var = tf.reduce_sum(m * (1 - m) / (S + 1), axis=1)
            loss = err + var

            kl_per_sample = self.dirichlet_kl_divergence(alpha)
            kl_div = tf.squeeze(kl_per_sample, axis=1)
            return tf.reduce_mean(loss + self.kl_weight * kl_div)
        return loss_fn

    def _build_model(self):
        input = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(self.num_classes, activation='softplus')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        class FisherWeightCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                self.parent = parent

            def on_epoch_begin(self, epoch, logs=None):
                self.parent.kl_weight = min(1.0, epoch / 10.0)

        fisher_callback = FisherWeightCallback(self)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose
        )
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose,
                                 callbacks=[fisher_callback, lr_scheduler])
        return history

    # get alphas
    def predict(self, inputs, verbose=0):
        evidence = self.model.predict(inputs, verbose=verbose)
        alpha = evidence + 1
        return alpha

    def predict_probs(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities

    # get softmax like probs (needed for foolbox integration only)
    def __call__(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities

class InformationEvidentialModel:
    def __init__(self, x_train, y_train, kl_weight=0.0001, learning_rate=0.01):
        self.x_train = x_train
        self.y_train = y_train
        self.input_shape = x_train.shape[1:]
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        self.fisher_weight = 0.001

        self.num_classes = y_train.shape[1]

        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=self.optimizer,
                           loss=self._fisher_evidential_loss(),
                           metrics=['accuracy'])

    def dirichlet_kl_divergence(self, alpha):
        alpha_shape = tf.shape(alpha)
        K = tf.cast(alpha_shape[1], alpha.dtype)
        alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
        kl = tf.math.lgamma(alpha_sum) - tf.math.lgamma(K)
        kl -= tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
        kl += tf.reduce_sum(
            (alpha - 1.0) * (tf.math.digamma(alpha) - tf.math.digamma(alpha_sum)),
            axis=1,
            keepdims=True
        )
        return kl

    def _fisher_evidential_loss(self):
        def loss_fn(y_true, outputs):
            evidence = outputs
            alpha = evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            m = alpha / S

            err = tf.reduce_sum((y_true - m) ** 2, axis=1)
            var = tf.reduce_sum(m * (1 - m) / (S + 1), axis=1)

            kl_per_sample = self.dirichlet_kl_divergence(alpha)
            kl_div = tf.squeeze(kl_per_sample, axis=1)

            trigamma_alpha = tf.math.polygamma(1.0, alpha)
            trigamma_alpha0 = tf.math.polygamma(1.0, S)
            fisher_term = tf.reduce_sum(tf.math.log(trigamma_alpha), axis=1) + \
                      tf.math.log(1.0 - tf.reduce_sum(trigamma_alpha0 / trigamma_alpha, axis=1))

            return tf.reduce_mean(err + var + (self.kl_weight * kl_div) - self.fisher_weight * fisher_term)
        return loss_fn

    def _build_model(self):
        input = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(self.num_classes, activation='softplus')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        class FisherWeightCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                self.parent = parent

            def on_epoch_begin(self, epoch, logs=None):
                self.parent.kl_weight = min(1.0, epoch / 10.0)
                self.parent.fisher_weight = min(1.0, epoch / 10.0)

        fisher_callback = FisherWeightCallback(self)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose
        )
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose,
                                 callbacks=[fisher_callback, lr_scheduler])
        return history

    # get alphas
    def predict(self, inputs, verbose=0):
        evidence = self.model.predict(inputs, verbose=verbose)
        alpha = evidence + 1
        return alpha

    def predict_probs(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities

    # get softmax like probs (needed for foolbox integration only)
    def __call__(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities
    
class SmoothedEvidentialModel:
    def __init__(self, x_train, y_train, learning_rate=0.01, sigma=0.01):
        self.x_train = x_train
        self.y_train = y_train
        self.input_shape = x_train.shape[1:]
        self.kl_weight = 1.0
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.num_samples = 50

        self.num_classes = y_train.shape[1]

        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=self.optimizer,
                           loss=self._evidential_loss(),
                           metrics=['accuracy'])

    def dirichlet_kl_divergence(self, alpha):
        alpha_shape = tf.shape(alpha)
        K = tf.cast(alpha_shape[1], alpha.dtype)
        alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
        kl = tf.math.lgamma(alpha_sum) - tf.math.lgamma(K)
        kl -= tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
        kl += tf.reduce_sum(
            (alpha - 1.0) * (tf.math.digamma(alpha) - tf.math.digamma(alpha_sum)),
            axis=1,
            keepdims=True
        )
        return kl

    def _evidential_loss(self):
        def loss_fn(y_true, outputs):
            evidence = outputs
            alpha = evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            m = alpha / S

            err = tf.reduce_sum((y_true - m) ** 2, axis=1)
            var = tf.reduce_sum(m * (1 - m) / (S + 1), axis=1)
            loss = err + var

            kl_per_sample = self.dirichlet_kl_divergence(alpha)
            kl_div = tf.squeeze(kl_per_sample, axis=1)
            return tf.reduce_mean(loss + self.kl_weight * kl_div)
        return loss_fn

    def _build_model(self):
        input = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(self.num_classes, activation='softplus')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        class FisherWeightCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                self.parent = parent

            def on_epoch_begin(self, epoch, logs=None):
                self.parent.kl_weight = min(1.0, epoch / 10.0)

        fisher_callback = FisherWeightCallback(self)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose
        )
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose,
                                 callbacks=[fisher_callback, lr_scheduler])
        return history

    def _generate_samples(self, inputs):
        """Generate perturbed samples from a Gaussian distribution."""
        batch_size, height, width, channels = inputs.shape
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=(self.num_samples, batch_size, height, width, channels))
        perturbed_samples = inputs[None, :, :, :, :] + noise
        perturbed_samples = np.reshape(perturbed_samples, (-1, height, width, channels))
        return perturbed_samples

    def predict(self, inputs, verbose=0):
        perturbed_samples = self._generate_samples(inputs)
        evidence = self.model.predict(perturbed_samples, verbose=0)
        evidence = np.reshape(evidence, (self.num_samples, -1, self.num_classes))
        alpha = evidence + 1
        smoothed_alphas = np.median(alpha, axis=0)
        return smoothed_alphas

    def predict_probs(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities

    # get softmax like probs (needed for foolbox integration only)
    def __call__(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities

class DensityAwareEvidentialModel:
    def __init__(self, x_train, y_train, kl_weight=0.0001, learning_rate=0.01):
        self.x_train = x_train
        self.y_train = y_train
        self.input_shape = x_train.shape[1:]
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate

        self.num_classes = y_train.shape[1]

        self.model, self.feature_extractor = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=self.optimizer,
                           loss=self._evidential_loss(),
                           metrics=['accuracy'])

        self.density_estimator = None
        self.density_min = None
        self.density_max = None

    def dirichlet_kl_divergence(self, alpha):
        alpha_shape = tf.shape(alpha)
        K = tf.cast(alpha_shape[1], alpha.dtype)
        alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
        kl = tf.math.lgamma(alpha_sum) - tf.math.lgamma(K)
        kl -= tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
        kl += tf.reduce_sum(
            (alpha - 1.0) * (tf.math.digamma(alpha) - tf.math.digamma(alpha_sum)),
            axis=1,
            keepdims=True
        )
        return kl

    def _evidential_loss(self):
        def loss_fn(y_true, outputs):
            logits = outputs
            evidence = tf.exp(logits)
            alpha = evidence
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            m = alpha / S
            err = tf.reduce_sum((y_true - m) ** 2, axis=1)
            var = tf.reduce_sum(m * (1 - m) / (S + 1), axis=1)
            loss = err + var
            kl_per_sample = self.dirichlet_kl_divergence(alpha)
            kl_div = tf.squeeze(kl_per_sample, axis=1)
            return tf.reduce_mean(loss + self.kl_weight * kl_div)
        return loss_fn

    def _build_model(self):
        input = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        features = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(features)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        logits = tf.keras.layers.Dense(self.num_classes, activation=None)(x)

        model = tf.keras.models.Model(inputs=input, outputs=logits)
        feature_extractor = tf.keras.models.Model(inputs=input, outputs=features)

        return model, feature_extractor

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        class FisherWeightCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                self.parent = parent

            def on_epoch_begin(self, epoch, logs=None):
                self.parent.kl_weight = min(1.0, epoch / 10.0)

        fisher_callback = FisherWeightCallback(self)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose
        )
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose,
                                 callbacks=[fisher_callback, lr_scheduler])

        self._fit_density_estimator()

        return history

    def _fit_density_estimator(self):
        features = self.feature_extractor.predict(self.x_train, verbose=0)

        self.density_estimator = sklearn.mixture.GaussianMixture(n_components=self.num_classes, covariance_type='full')
        self.density_estimator.fit(features)

        log_densities = self.density_estimator.score_samples(features)
        self.density_min = np.min(log_densities)
        self.density_max = np.max(log_densities)

    def _compute_normalised_density(self, inputs):
        features = self.feature_extractor.predict(inputs, verbose=0)
        log_density = self.density_estimator.score_samples(features)

        # normalise log density to [0, 1]
        normalised = (log_density - self.density_min) / (self.density_max - self.density_min)
        normalised = np.clip(normalised, 0.0, 1.0)
        return normalised[:, np.newaxis]  # add axis for broadcasting

    def predict(self, inputs, apply_density_scaling=True, verbose=0):
        logits = self.model(inputs, training=False)

        if apply_density_scaling:
            s = self._compute_normalised_density(inputs)
            logits = logits * tf.convert_to_tensor(s, dtype=logits.dtype)

        evidence = tf.exp(logits)
        alpha = evidence
        return alpha

    def predict_probs(self, inputs, apply_density_scaling=True):
        alpha = self.predict(inputs, apply_density_scaling=apply_density_scaling)
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities

    def __call__(self, inputs):
        return self.predict_probs(inputs)

class EvidentialPlusMetaModel:
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

    def dirichlet_kl_divergence(self, alpha):
        alpha_shape = tf.shape(alpha)
        K = tf.cast(alpha_shape[1], alpha.dtype)
        alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
        kl = tf.math.lgamma(alpha_sum) - tf.math.lgamma(K)
        kl -= tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
        kl += tf.reduce_sum(
            (alpha - 1.0) * (tf.math.digamma(alpha) - tf.math.digamma(alpha_sum)),
            axis=1,
            keepdims=True
        )
        return kl

    def _evidential_loss(self):
        def loss_fn(y_true, outputs):
            evidence = outputs
            alpha = evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            m = alpha / S

            err = tf.reduce_sum((y_true - m) ** 2, axis=1)
            var = tf.reduce_sum(m * (1 - m) / (S + 1), axis=1)
            loss = err + var

            kl_per_sample = self.dirichlet_kl_divergence(alpha)
            kl_div = tf.squeeze(kl_per_sample, axis=1)
            return tf.reduce_mean(loss + self.kl_weight * kl_div)
        return loss_fn

    def _build_model(self):
        input = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(self.num_classes, activation='softplus')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        class FisherWeightCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                self.parent = parent

            def on_epoch_begin(self, epoch, logs=None):
                self.parent.kl_weight = min(1.0, epoch / 10.0)

        fisher_callback = FisherWeightCallback(self)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose
        )
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose,
                                 callbacks=[fisher_callback, lr_scheduler])
        return history

    def apply_random_transformation(self, inputs):
        transformations = ['rotate', 'shift', 'add_noise']
        transform = random.choice(transformations)
        inputs_transformed = inputs

        if transform == 'rotate':
            angle = random.uniform(-15, 15)
            inputs_transformed = scipy.ndimage.rotate(inputs, angle, reshape=False, mode='nearest')
        elif transform == 'shift':
            shift_val = random.randint(-2, 2)
            inputs_transformed = np.roll(inputs, shift=shift_val, axis=0)
        elif transform == 'add_noise':
            noise = np.random.normal(0, 0.01, inputs.shape)
            inputs_transformed = inputs + noise

        inputs_transformed = np.clip(inputs_transformed, 0, 1)
        return inputs_transformed

    def apply_metamorphic_transformations(self, inputs, num_transforms=5):
        transformed_images = [np.array(inputs)]
        for _ in range(num_transforms - 1):
            transformed_batch = np.array([self.apply_random_transformation(img) for img in inputs])
            transformed_images.append(transformed_batch)
        return np.stack(transformed_images, axis=0)

    def predict_with_metamorphic_transforms(self, inputs, num_transforms=5):
        original_evidence = self.model.predict(inputs, verbose=0)
        transformed_inputs = self.apply_metamorphic_transformations(inputs, num_transforms)
        batch_size = inputs.shape[0]
        flat_transformed_inputs = tf.reshape(transformed_inputs, [-1, *self.input_shape])
        evidences = self.model.predict(flat_transformed_inputs, verbose=0)
        evidences = tf.reshape(evidences, [num_transforms, batch_size, self.num_classes])
        evidences = tf.transpose(evidences, [1, 2, 0])

        mean_evidence = tf.reduce_mean(evidences, axis=-1)
        return original_evidence, mean_evidence.numpy()

    def predict(self, inputs, num_transforms=5, verbose=0, for_threshold=False):
        original_evidence, averaged_evidence = self.predict_with_metamorphic_transforms(inputs, num_transforms)
        if for_threshold:
            alpha = original_evidence + 1
        else:
            alpha = averaged_evidence + 1
        return alpha

    def predict_probs(self, inputs, num_transforms=5, verbose=0, for_threshold=False):
        original_evidence, averaged_evidence = self.predict_with_metamorphic_transforms(inputs, num_transforms)
        if for_threshold:
            original_evidence = tf.convert_to_tensor(original_evidence, dtype=tf.float32)
            alpha = original_evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            probabilities = alpha / S
        else:
            averaged_evidence = tf.convert_to_tensor(averaged_evidence, dtype=tf.float32)
            alpha = averaged_evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            probabilities = alpha / S
        return probabilities

    def __call__(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities
    
class EvidentialPlusMcModel:
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

    def dirichlet_kl_divergence(self, alpha):
        alpha_shape = tf.shape(alpha)
        K = tf.cast(alpha_shape[1], alpha.dtype)
        alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
        kl = tf.math.lgamma(alpha_sum) - tf.math.lgamma(K)
        kl -= tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
        kl += tf.reduce_sum(
            (alpha - 1.0) * (tf.math.digamma(alpha) - tf.math.digamma(alpha_sum)),
            axis=1,
            keepdims=True
        )
        return kl

    def _evidential_loss(self):
        def loss_fn(y_true, outputs):
            evidence = outputs
            alpha = evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            m = alpha / S

            err = tf.reduce_sum((y_true - m) ** 2, axis=1)
            var = tf.reduce_sum(m * (1 - m) / (S + 1), axis=1)
            loss = err + var

            kl_per_sample = self.dirichlet_kl_divergence(alpha)
            kl_div = tf.squeeze(kl_per_sample, axis=1)
            return tf.reduce_mean(loss + self.kl_weight * kl_div)
        return loss_fn

    def _build_model(self):
        input = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(self.num_classes, activation='softplus')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        class FisherWeightCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                self.parent = parent

            def on_epoch_begin(self, epoch, logs=None):
                self.parent.kl_weight = min(1.0, epoch / 10.0)

        fisher_callback = FisherWeightCallback(self)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose
        )
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose,
                                 callbacks=[fisher_callback, lr_scheduler])
        return history

    def predict_with_mc_dropout(self, inputs, num_passes=5, delta=1.0, alpha=1.5):
        original_evidence = self.model.predict(inputs, verbose=0)
        batch_size = inputs.shape[0]
        mc_evidences = []
        for _ in range(num_passes):
            ev = self.model(inputs, training=True)
            mc_evidences.append(ev)
        mc_evidences = tf.stack(mc_evidences, axis=0)
        mc_evidences = tf.transpose(mc_evidences, [1, 2, 0])
        adjusted_evidence = np.mean(mc_evidences.numpy(), axis=2)
        return original_evidence, np.array(adjusted_evidence)

    def predict(self, inputs, num_transforms=5, verbose=0, for_threshold=False):
        original_evidence, averaged_evidence = self.predict_with_mc_dropout(inputs, num_transforms)
        if for_threshold:
            alpha = original_evidence + 1
        else:
            alpha = averaged_evidence + 1
        return alpha

    def predict_probs(self, inputs, num_transforms=5, verbose=0, for_threshold=False):
        original_evidence, averaged_evidence = self.predict_with_mc_dropout(inputs, num_transforms)
        if for_threshold:
            original_evidence = tf.convert_to_tensor(original_evidence, dtype=tf.float32)
            alpha = original_evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            probabilities = alpha / S
        else:
            averaged_evidence = tf.convert_to_tensor(averaged_evidence, dtype=tf.float32)
            alpha = averaged_evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            probabilities = alpha / S
        return probabilities

    def __call__(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities
    
class ConflictingEvidentialMetaModel:
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

    def dirichlet_kl_divergence(self, alpha):
        alpha_shape = tf.shape(alpha)
        K = tf.cast(alpha_shape[1], alpha.dtype)
        alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
        kl = tf.math.lgamma(alpha_sum) - tf.math.lgamma(K)
        kl -= tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
        kl += tf.reduce_sum(
            (alpha - 1.0) * (tf.math.digamma(alpha) - tf.math.digamma(alpha_sum)),
            axis=1,
            keepdims=True
        )
        return kl

    def _evidential_loss(self):
        def loss_fn(y_true, outputs):
            evidence = outputs
            alpha = evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            m = alpha / S

            err = tf.reduce_sum((y_true - m) ** 2, axis=1)
            var = tf.reduce_sum(m * (1 - m) / (S + 1), axis=1)
            loss = err + var

            kl_per_sample = self.dirichlet_kl_divergence(alpha)
            kl_div = tf.squeeze(kl_per_sample, axis=1)
            return tf.reduce_mean(loss + self.kl_weight * kl_div)
        return loss_fn

    def _build_model(self):
        input = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(self.num_classes, activation='softplus')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        class FisherWeightCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                self.parent = parent

            def on_epoch_begin(self, epoch, logs=None):
                self.parent.kl_weight = min(1.0, epoch / 10.0)

        fisher_callback = FisherWeightCallback(self)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose
        )
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose,
                                 callbacks=[fisher_callback, lr_scheduler])
        return history

    def apply_random_transformation(self, inputs):
        transformations = ['rotate', 'shift', 'add_noise']
        transform = random.choice(transformations)
        inputs_transformed = inputs

        if transform == 'rotate':
            angle = random.uniform(-15, 15)  # Generate random angle between -15° and +15°
            inputs_transformed = scipy.ndimage.rotate(inputs, angle, reshape=False, mode='nearest')
        elif transform == 'shift':
            shift_val = random.randint(-2, 2)
            inputs_transformed = np.roll(inputs, shift=shift_val, axis=0)
        elif transform == 'add_noise':
            noise = np.random.normal(0, 0.01, inputs.shape)
            inputs_transformed = inputs + noise

        inputs_transformed = np.clip(inputs_transformed, 0, 1)
        return inputs_transformed

    def apply_metamorphic_transformations(self, inputs, num_transforms=5):
        transformed_images = [np.array(inputs)]
        for _ in range(num_transforms - 1):
            transformed_batch = np.array([self.apply_random_transformation(img) for img in inputs])
            transformed_images.append(transformed_batch)
        return np.stack(transformed_images, axis=0)

    def calculate_kintra(self, evidence):
        evidence = np.array(evidence)
        n_classes, n_runs = evidence.shape
        epsilon = 1e-8
        intra_class_normalized_std = []
        for i in range(n_classes):
            class_evidence = evidence[i]
            std_dev = np.std(class_evidence)
            mean_evidence = np.mean(class_evidence)
            normalized_std = std_dev / (mean_evidence + epsilon)
            intra_class_normalized_std.append(normalized_std)
        k_intra = np.mean(intra_class_normalized_std)
        return k_intra

    def calculate_krun_inter(self, evidence, alpha):
        evidence = np.array(evidence)
        n_classes, n_runs = evidence.shape
        krun_inter_values = []
        for run in range(n_runs):
            pairwise_conflicts = []
            e_values = evidence[:, run]
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    min_e = min(e_values[i], e_values[j])
                    max_e = max(e_values[i], e_values[j])
                    if max_e == 0:
                        continue
                    term = (min_e / max_e) * (min_e / np.sum(e_values)) * 2
                    pairwise_conflicts.append(term)
            if pairwise_conflicts:
                sum_power_means = sum([conflict ** 2 for conflict in pairwise_conflicts]) ** (1 / 2)
                scaled_sum_power_means = 1 - np.exp(-alpha * sum_power_means)
                krun_inter_values.append(scaled_sum_power_means)
            else:
                krun_inter_values.append(0)
        return np.mean(krun_inter_values)

    def compute_K_total(self, K_inter, K_intra):
        lambda_penalty = 0.5
        penalty_grid = lambda_penalty * (K_inter - K_intra)**2
        K_total = K_inter + K_intra - K_inter * K_intra - penalty_grid
        K_total = np.clip(K_total, 0, 1)
        return K_total

    def process_evidence(self, evidence, alpha):
        K_intra = self.calculate_kintra(evidence)
        K_inter = self.calculate_krun_inter(evidence, alpha)
        K_total = self.compute_K_total(K_inter, K_intra)
        return K_total, K_inter, K_intra

    def predict_with_metamorphic_transforms(self, inputs, num_transforms=5, delta=1.0, alpha=1.5):
        original_evidence = self.model.predict(inputs, verbose=0)
        transformed_inputs = self.apply_metamorphic_transformations(inputs, num_transforms)
        batch_size = inputs.shape[0]
        flat_transformed_inputs = tf.reshape(transformed_inputs, [-1, *self.input_shape])
        evidences = self.model.predict(flat_transformed_inputs, verbose=0)
        evidences = tf.reshape(evidences, [num_transforms, batch_size, self.num_classes])
        evidences = tf.transpose(evidences, [1, 2, 0])

        adjusted_evidence = []
        for sample_evidences in evidences.numpy():
            K_total, K_inter, K_intra = self.process_evidence(sample_evidences, alpha)
            mean_evidence = np.mean(sample_evidences, axis=1)
            scaling_factor = np.exp(-K_total * delta)
            adjusted_ev = mean_evidence * scaling_factor
            adjusted_evidence.append(adjusted_ev)
        return original_evidence, np.array(adjusted_evidence)

    def predict(self, inputs, num_transforms=5, verbose=0, for_threshold=False):
        original_evidence, averaged_evidence = self.predict_with_metamorphic_transforms(inputs, num_transforms)
        if for_threshold:
            alpha = original_evidence + 1
        else:
            alpha = averaged_evidence + 1
        return alpha

    def predict_probs(self, inputs, num_transforms=5, verbose=0, for_threshold=False):
        original_evidence, averaged_evidence = self.predict_with_metamorphic_transforms(inputs, num_transforms)
        if for_threshold:
            original_evidence = tf.convert_to_tensor(original_evidence, dtype=tf.float32)
            alpha = original_evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            probabilities = alpha / S
        else:
            averaged_evidence = tf.convert_to_tensor(averaged_evidence, dtype=tf.float32)
            alpha = averaged_evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            probabilities = alpha / S
        return probabilities

    def __call__(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities
    
class ConflictingEvidentialMcModel:
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

    def dirichlet_kl_divergence(self, alpha):
        alpha_shape = tf.shape(alpha)
        K = tf.cast(alpha_shape[1], alpha.dtype)
        alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
        kl = tf.math.lgamma(alpha_sum) - tf.math.lgamma(K)
        kl -= tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
        kl += tf.reduce_sum(
            (alpha - 1.0) * (tf.math.digamma(alpha) - tf.math.digamma(alpha_sum)),
            axis=1,
            keepdims=True
        )
        return kl

    def _evidential_loss(self):
        def loss_fn(y_true, outputs):
            evidence = outputs
            alpha = evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            m = alpha / S

            err = tf.reduce_sum((y_true - m) ** 2, axis=1)
            var = tf.reduce_sum(m * (1 - m) / (S + 1), axis=1)
            loss = err + var

            kl_per_sample = self.dirichlet_kl_divergence(alpha)
            kl_div = tf.squeeze(kl_per_sample, axis=1)
            return tf.reduce_mean(loss + self.kl_weight * kl_div)
        return loss_fn

    def _build_model(self):
        input = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(self.num_classes, activation='softplus')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        class FisherWeightCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                self.parent = parent

            def on_epoch_begin(self, epoch, logs=None):
                self.parent.kl_weight = min(1.0, epoch / 10.0)

        fisher_callback = FisherWeightCallback(self)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=verbose
        )
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose,
                                 callbacks=[fisher_callback, lr_scheduler])
        return history

    def calculate_kintra(self, evidence):
        evidence = np.array(evidence)
        n_classes, n_runs = evidence.shape
        epsilon = 1e-8
        intra_class_normalized_std = []
        for i in range(n_classes):
            class_evidence = evidence[i]
            std_dev = np.std(class_evidence)
            mean_evidence = np.mean(class_evidence)
            normalized_std = std_dev / (mean_evidence + epsilon)
            intra_class_normalized_std.append(normalized_std)
        k_intra = np.mean(intra_class_normalized_std)
        return k_intra

    def calculate_krun_inter(self, evidence, alpha):
        evidence = np.array(evidence)
        n_classes, n_runs = evidence.shape
        krun_inter_values = []
        for run in range(n_runs):
            pairwise_conflicts = []
            e_values = evidence[:, run]
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    min_e = min(e_values[i], e_values[j])
                    max_e = max(e_values[i], e_values[j])
                    if max_e == 0:
                        continue
                    term = (min_e / max_e) * (min_e / np.sum(e_values)) * 2
                    pairwise_conflicts.append(term)
            if pairwise_conflicts:
                sum_power_means = sum([conflict ** 2 for conflict in pairwise_conflicts]) ** (1 / 2)
                scaled_sum_power_means = 1 - np.exp(-alpha * sum_power_means)
                krun_inter_values.append(scaled_sum_power_means)
            else:
                krun_inter_values.append(0)
        return np.mean(krun_inter_values)

    def compute_K_total(self, K_inter, K_intra):
        lambda_penalty = 0.5
        penalty_grid = lambda_penalty * (K_inter - K_intra)**2
        K_total = K_inter + K_intra - K_inter * K_intra - penalty_grid
        K_total = np.clip(K_total, 0, 1)
        return K_total

    def process_evidence(self, evidence, alpha):
        K_intra = self.calculate_kintra(evidence)
        K_inter = self.calculate_krun_inter(evidence, alpha)
        K_total = self.compute_K_total(K_inter, K_intra)
        return K_total, K_inter, K_intra

    def predict_with_mc_dropout(self, inputs, num_passes=5, delta=1.0, alpha=1.5):
        original_evidence = self.model.predict(inputs, verbose=0)
        batch_size = inputs.shape[0]
        mc_evidences = []
        for _ in range(num_passes):
            ev = self.model(inputs, training=True)
            mc_evidences.append(ev)
        mc_evidences = tf.stack(mc_evidences, axis=0)
        mc_evidences = tf.transpose(mc_evidences, [1, 2, 0])
        adjusted_evidence = []
        for sample_evidences in mc_evidences.numpy():
            K_total, K_inter, K_intra = self.process_evidence(sample_evidences, alpha)
            mean_evidence = np.mean(sample_evidences, axis=1)
            scaling_factor = np.exp(-K_total * delta)
            adjusted_ev = mean_evidence * scaling_factor
            adjusted_evidence.append(adjusted_ev)
        return original_evidence, np.array(adjusted_evidence)

    def predict(self, inputs, num_transforms=5, verbose=0, for_threshold=False):
        original_evidence, averaged_evidence = self.predict_with_mc_dropout(inputs, num_transforms)
        if for_threshold:
            alpha = original_evidence + 1
        else:
            alpha = averaged_evidence + 1
        return alpha

    def predict_probs(self, inputs, num_transforms=5, verbose=0, for_threshold=False):
        original_evidence, averaged_evidence = self.predict_with_mc_dropout(inputs, num_transforms)
        if for_threshold:
            original_evidence = tf.convert_to_tensor(original_evidence, dtype=tf.float32)
            alpha = original_evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            probabilities = alpha / S
        else:
            averaged_evidence = tf.convert_to_tensor(averaged_evidence, dtype=tf.float32)
            alpha = averaged_evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            probabilities = alpha / S
        return probabilities

    def __call__(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities
