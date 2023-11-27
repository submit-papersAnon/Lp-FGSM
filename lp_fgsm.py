# Imports
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model


# $l^p$-FGSM
class Lp_FGSM(Model):
    """
    Lp Fast Gradient Sign Method (FGSM) adversarial training model.

    This class extends the Keras Model class, enabling the creation of adversarial examples
    based on an Lp norm during training.

    Attributes:
        base_model: The underlying model for making predictions.
        p (float): The norm degree for Lp norm.
        eps (float): Maximum perturbation allowed for adversarial examples.
        cce (tf.keras.losses.Loss): Custom categorical cross-entropy loss function.
        add_noise (bool): Flag to add noise to the inputs for adversarial generation.
        vareps (float): A small constant for numerical stability in norm calculations.
    """

    def __init__(self, base_model, p=64.0, eps=8.0, cce=None, add_noise=True, vareps=1e-12, *args, **kwargs):
        super(Lp_FGSM, self).__init__(*args, **kwargs)
        self.base_model = base_model  # The underlying model for predictions
        self.p = tf.constant(p, dtype=tf.float32)
        self.q = self.p / (self.p - 1.0)
        self.eps = tf.constant(eps / 255.0, dtype=tf.float32)
        self.vareps = tf.constant(vareps, dtype=tf.float32)
        self.add_noise = add_noise
        self.cce = cce if cce is not None else tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


    @tf.function(jit_compile=True)
    def call(self, inputs, training=True):
        return self.base_model(inputs, training=training)

    @tf.function(jit_compile=True)
    def train_step(self, data):
        """
        Custom training step for the Lp_FGSM adversarial training.

        Args:
            data: Tuple of (input data, labels).

        Returns:
            Dictionary mapping metric names to their current value.
        """
        x, y = data  # Unpack the data
        probs = self.base_model(x, training=True)
        # Generate adversarial examples
        if self.add_noise:
            # Add random noise and compute the Lp norm perturbation
            x_rnd = tf.random.uniform(tf.shape(x), minval=-1.0, maxval=1.0)
            x_adv = x + self.eps * x_rnd
            Ups_ = tf.pow(tf.reduce_sum(tf.pow(tf.abs(x_rnd), self.q), axis=[1, 2, 3], keepdims=True), 1.0 / self.q)
            Upsilon = tf.pow(self.vareps + tf.abs(x_rnd) / (Ups_), self.q - 1.0)
            x_rnd = tf.sign(x_rnd) * Upsilon
            x_aug = x + self.eps * x_rnd
        else:
            x_aug = x
            x_adv = x

        # Gradient computation for the augmented data
        with tf.GradientTape(watch_accessed_variables=False, persistent=False) as inner_tape:
            inner_tape.watch(x_aug)
            probs_aug = self.base_model(x_aug, training=True)
            loss_sum = self.cce(y, probs_aug)
        dlx = inner_tape.gradient(loss_sum, x_aug)
        dlxq = tf.pow(tf.reduce_sum(tf.pow(tf.abs(dlx), self.q), axis=[1, 2, 3], keepdims=True), 1.0 / self.q)
        Upsilon = tf.pow(self.vareps + tf.abs(dlx) / (dlxq), self.q - 1.0)
        dlxn = tf.sign(dlx) * Upsilon
        x_adv += self.eps * dlxn

        # Training step for the adversarial data
        with tf.GradientTape(watch_accessed_variables=False, persistent=False) as tape:
            tape.watch(self.base_model.trainable_variables)
            probs_adv = self.base_model(x_adv, training=True)
            losses_adv = self.cce(y, probs_adv)
            loss_total = tf.reduce_mean(losses_adv)

        grads = tape.gradient(loss_total, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        self.compiled_metrics.update_state(y, probs)
        return {m.name: m.result() for m in self.metrics}
