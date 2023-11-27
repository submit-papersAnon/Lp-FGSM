# Imports
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import Progbar


# @title PGD attack

def clip_epsilon(tensor, epsilon):
    """
    Clips the input tensor values to be within the range [-epsilon, epsilon].

    :param tensor: Input tensor.
    :param epsilon: Maximum allowed absolute value for elements in the tensor.
    :return: Tensor with values clipped to the specified range.
    """
    return tf.clip_by_value(tensor, -epsilon, epsilon)

def pgd_attack(model, x, y, epsilon, alpha, attack_iters, clip_min=0.0, clip_max=1.0):
    """
    Performs the Projected Gradient Descent (PGD) attack on a batch of images.

    Args:
        model: The neural network model to attack.
        x: Input images (batch).
        y: True labels for x.
        epsilon: The maximum perturbation allowed (L-infinity norm).
        alpha: Step size for each iteration of the attack.
        attack_iters: Number of iterations for the attack.
        clip_min: Minimum pixel value after perturbation.
        clip_max: Maximum pixel value after perturbation.

    Returns:
        A batch of adversarial images generated from the input images.
    """

    # Normalize epsilon and alpha according to the image scale (0-255)
    epsilon = epsilon / 255.0
    alpha = alpha / 255.0

    # Initialize adversarial images with random perturbations
    adv_x = x + tf.random.uniform(tf.shape(x), minval=-epsilon, maxval=epsilon)
    adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)  # Ensure they stay within valid pixel range

    # Iteratively apply the PGD attack
    for _ in tf.range(attack_iters):
        with tf.GradientTape() as tape:
            tape.watch(adv_x)  # Watch the adversarial examples for gradient computation
            logits = model(adv_x)  # Compute the model's output on the adversarial examples
            loss = model.compiled_loss(y, logits)  # Calculate loss

        # Compute gradients of the loss w.r.t. adversarial examples
        gradients = tape.gradient(loss, adv_x)
        
        # Update adversarial examples using the sign of the gradients
        adv_x = adv_x + alpha * tf.sign(gradients)

        # Clip the adversarial examples to stay within epsilon-ball and valid pixel range
        adv_x = x + clip_epsilon(adv_x-x, epsilon)
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x





def evaluate_pgd(model, dataset, epsilon=8, alpha=2, attack_iters=50, restarts=1, batch_size=64):
    adversarial_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    total_batches = sum(1 for _ in dataset.batch(batch_size))
    print(f"\nEvaluating PGD-{attack_iters}-{restarts} on {total_batches} batches...")

    progbar = tf.keras.utils.Progbar(total_batches * restarts)
    batch_count = 0

    # Evaluate clean accuracy using model.evaluate
    clean_results = model.evaluate(dataset.batch(batch_size), verbose=0)
    standard_accuracy = clean_results[1]  # Assuming the accuracy is the second metric

    for batch_num, (x, y) in enumerate(dataset.batch(batch_size)):
        best_adv_x = x
        for restart_num in range(restarts):
            adv_x = pgd_attack(model, x, y, epsilon, alpha, attack_iters)
            
            # Keep adversarial examples where model predictions are incorrect
            incorrect_preds = tf.argmax(model(adv_x, training=False), axis=1) != tf.argmax(y, axis=1)
            incorrect_preds = tf.reshape(incorrect_preds, [-1, 1, 1, 1])
            best_adv_x = tf.where(incorrect_preds, adv_x, best_adv_x)
            
            progbar.update(batch_count + 1)
            batch_count += 1

        logits_adv = model(best_adv_x, training=False)
        adversarial_acc_metric.update_state(y, logits_adv)

    adversarial_accuracy = adversarial_acc_metric.result().numpy()

    print(f"PGD-{attack_iters}-{restarts} Evaluation complete. \nValidation Accuracy: {100.0*standard_accuracy:.2f}%, PGD-{attack_iters}-{restarts} Adversarial Accuracy: {100.0*adversarial_accuracy:.2f}%")
    return {"standard_accuracy": standard_accuracy, "adversarial_accuracy": adversarial_accuracy}



