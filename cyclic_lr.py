# @title Imports
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras




# @title Cyclic learning rate

class CyclicLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cyclic learning rate scheduler.

    This scheduler varies the learning rate between a maximum and a minimum value in a cosine
    wave pattern. The learning rate starts at `lr_max` and gradually decreases to `lr_min`,
    then goes back to `lr_max` in a cyclic fashion.

    Args:
        lr_max (float): The maximum learning rate.
        lr_min (float): The minimum learning rate.
        nb_epochs (int): The number of epochs over which the learning rate cycles.
    """
    def __init__(self, lr_max, lr_min, nb_epochs):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.nb_epochs = nb_epochs

    def __call__(self, step):
        """
        Calculate the learning rate for a given step.
        Args:
            step (int): The current training step.

        Returns:
            float: The calculated learning rate.
        """
        # Calculate the current epoch based on the step
        epoch = tf.cast(step // self.nb_epochs, dtype=tf.float32)
        
        # Calculate the position in the current cycle
        cycle = tf.constant(np.pi, dtype=tf.float32) * epoch / tf.cast(self.nb_epochs, dtype=tf.float32)
        
        # Calculate and return the cyclic learning rate
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + tf.math.cos(cycle))


#Usage Eaxmple

## Parameters for the cyclic learning rate
#lr_max = 0.2
#lr_min = 0.001
#nb_epochs = epochs

# Create an instance of the CyclicLR scheduler
#cyclic_lr_schedule = CyclicLR(lr_max, lr_min, nb_epochs)
