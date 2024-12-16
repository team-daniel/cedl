import scipy.ndimage
import tensorflow as tf
import numpy as np
import sklearn
import random
import scipy

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
        x = tf.keras.layers.Conv2D(6, (5, 5), activation='tanh', padding='same')(input)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(16, (5, 5), activation='tanh')(x) 
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='tanh')(x)
        x = tf.keras.layers.Dense(84, activation='tanh')(x)
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
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
    
    def predict_probs(self, inputs):
        evidence = self.model(inputs)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities = alpha / S
        return probabilities

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
        x = tf.keras.layers.Conv2D(6, (5, 5), activation='tanh', padding='same')(input)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(16, (5, 5), activation='tanh')(x) 
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='tanh')(x)
        x = tf.keras.layers.Dense(84, activation='tanh')(x)
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
    
class EvidentialPlusModel:
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
        x = tf.keras.layers.Conv2D(6, (5, 5), activation='tanh', padding='same')(input)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(16, (5, 5), activation='tanh')(x) 
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='tanh')(x)
        x = tf.keras.layers.Dense(84, activation='tanh')(x)
        output = tf.keras.layers.Dense(self.num_classes, activation='softplus')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose)
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
    
class ConflictingEvidentialModel:
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
        x = tf.keras.layers.Conv2D(6, (5, 5), activation='tanh', padding='same')(input)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(16, (5, 5), activation='tanh')(x) 
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation='tanh')(x)
        x = tf.keras.layers.Dense(84, activation='tanh')(x)
        output = tf.keras.layers.Dense(self.num_classes, activation='softplus')(x)
        return tf.keras.models.Model(inputs=input, outputs=output)

    def train(self, batch_size=128, epochs=100, validation_split=0.2, verbose=0):
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=verbose)
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
    

