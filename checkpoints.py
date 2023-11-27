# Imports
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from attacks import evaluate_pgd  # Importing evaluate_pgd from attack.py

# Checkpoints


class AdversarialCheckpoint(Callback):
    def __init__(self, dataset, epsilon=8.0, alpha=2.0, attack_iters=50, restarts=1, adv_batch_size=64):
        """
        Callback to evaluate model performance on adversarial examples after training.

        :param dataset: tf.data.Dataset for evaluation.
        :param epsilon: Maximum perturbation for PGD attack.
        :param alpha: Step size for PGD attack.
        :param attack_iters: Number of iterations for PGD attack.
        :param restarts: Number of restarts for PGD attack.
        :param batch_size: Batch size for evaluation.
        """
        super(AdversarialCheckpoint, self).__init__()
        self.dataset = dataset
        self.epsilon = epsilon
        self.alpha = alpha
        self.attack_iters = attack_iters
        self.restarts = restarts
        self.adv_batch_size = adv_batch_size
        self.adv_accuracy = 0.0
        self.test_accuracy = 0.0

    def on_train_end(self, logs=None):
        """
        Called at the end of training. Evaluates the model on both clean and adversarial examples.
        """
        # Evaluate the model on adversarial examples using PGD with multiple restarts
        accuracies = evaluate_pgd(
            model=self.model, dataset=self.dataset, epsilon=self.epsilon, alpha=self.alpha, 
            attack_iters=self.attack_iters, restarts=self.restarts, batch_size=self.adv_batch_size
        )

        self.adv_accuracy = accuracies["adversarial_accuracy"]
        self.test_accuracy = accuracies["standard_accuracy"]



class AdversarialCheckpoint_Epochs(Callback):
    def __init__(self, dataset, epsilon=8.0, alpha=2.0, attack_iters=50, restarts=1, adv_batch_size=64):
        """
        Callback to evaluate model performance on adversarial examples after each epoch.

        :param dataset: tf.data.Dataset for evaluation.
        :param epsilon: Maximum perturbation for PGD attack.
        :param alpha: Step size for PGD attack.
        :param attack_iters: Number of iterations for PGD attack.
        :param restarts: Number of restarts for PGD attack.
        :param batch_size: Batch size for evaluation.
        """
        super(AdversarialCheckpoint_Epochs, self).__init__()
        self.dataset = dataset
        self.epsilon = epsilon
        self.alpha = alpha
        self.attack_iters = attack_iters
        self.restarts = restarts
        self.adv_batch_size = adv_batch_size

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch. Evaluates the model on both clean and adversarial examples.
        """
        # Evaluate the model on adversarial examples using PGD with multiple restarts
        accuracies = evaluate_pgd(
            model=self.model, dataset=self.dataset, epsilon=self.epsilon, alpha=self.alpha, 
            attack_iters=self.attack_iters, restarts=self.restarts, batch_size=self.adv_batch_size
        )

        self.adv_accuracy = accuracies["adversarial_accuracy"]
        self.test_accuracy = accuracies["standard_accuracy"]