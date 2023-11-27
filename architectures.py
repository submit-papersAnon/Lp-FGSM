# Imports
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, 
                                     Activation, Add, GlobalAveragePooling2D, 
                                     Dense, MaxPooling2D, AveragePooling2D, Dropout, Flatten)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# PreActResNet18


def preact_res_block(x, filters, activation='relu', kernel_size=(3, 3), stride=1, weight_decay=0.0, dropout_rate=0.0):
    """
    Creates a pre-activation residual block for PreActResNet.

    Args:
        x: Input tensor or layer.
        filters: Number of filters for the convolution layers.
        activation: Activation function to use.
        kernel_size: Size of the convolution kernel.
        stride: Stride size for the convolution.
        weight_decay: L2 regularization factor.
        dropout_rate: Dropout rate.

    Returns:
        A tensor representing the output of the residual block.
    """
    shortcut = x

    # Applying batch normalization, activation, and convolution twice
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Conv2D(filters, kernel_size, strides=stride if _ == 0 else 1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        if dropout_rate > 0.0: 
            x = Dropout(dropout_rate)(x)

    # Adjusting shortcut path for dimensionality matching (if needed)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(shortcut)

    x = Add()([shortcut, x])  # Skip connection (element-wise addition)
    return x

def PreActResNet18(input_shape, num_classes=10, activation='relu', weight_decay=0.0, dropout_rate=0.0):
    """
    Constructs a PreActResNet18 model.

    Args:
        input_shape: Shape of the input data.
        num_classes: Number of classes for the output layer.
        activation: Activation function to use in the blocks.
        weight_decay: L2 regularization factor.
        dropout_rate: Dropout rate.

    Returns:
        A PreActResNet18 model.
    """
    input = Input(input_shape)
    x = Conv2D(64, (7, 7), strides=2, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(2.0 * (input - 0.5))
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Constructing ResNet blocks with specified activation function
    for filters, repetitions, use_strided_conv in zip([64, 128, 256, 512], [2, 2, 2, 2], [False, True, True, True]):
        for i in range(repetitions):
            x = preact_res_block(x, filters, activation, stride=2 if i == 0 and use_strided_conv else 1, weight_decay=weight_decay, dropout_rate=dropout_rate)

    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0.0: 
        x = Dropout(dropout_rate)(x)
    x = Dense(num_classes, kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)  # Softmax activation for classification

    return Model(input, x)



# WideResNet


def initial_conv(input, weight_decay, activation='relu'):
    """
    Initial convolutional layer for WideResNet.
    
    Args:
        input: Input tensor or layer.
        weight_decay: L2 regularization factor.
        activation: Activation function to use.

    Returns:
        Tensor after applying Conv2D, BatchNormalization, and the specified Activation.
    """
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), use_bias=False)(input)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation(activation)(x)
    return x

def expand_conv(init, base, k, strides=(1, 1), weight_decay=0.0, activation='relu'):
    """
    Expanding convolution layer in WideResNet. Increases the dimensionality of the input tensor.

    Args:
        init: Initial tensor or input layer.
        base: Number of base filters.
        k: Width factor for scaling the number of filters.
        strides: Convolution strides.
        weight_decay: L2 regularization factor.
        activation: Activation function to use.

    Returns:
        Tensor after applying Conv2D, BatchNormalization, and Activation.
    """
    x = Conv2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), use_bias=False)(init)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation(activation)(x)

    x = Conv2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), use_bias=False)(x)

    # Creating a shortcut path to implement the skip connection
    skip = Conv2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), use_bias=False)(init)
    m = Add()([x, skip])  # Adding the skip connection
    return m

def conv_block(input, k, base, dropout, weight_decay, activation='relu'):
    """
    Standard convolutional block for WideResNet.

    Args:
        input: Input tensor or layer.
        k: Width factor for scaling the number of filters.
        base: Number of base filters.
        dropout: Dropout rate.
        weight_decay: L2 regularization factor.
        activation: Activation function to use.

    Returns:
        Tensor after applying Conv2D, BatchNormalization, Activation, and optionally Dropout.
    """
    init = input
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation(activation)(x)
    x = Conv2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), use_bias=False)(x)

    if dropout > 0.0: 
        x = Dropout(dropout)(x)

    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation(activation)(x)
    x = Conv2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), use_bias=False)(x)

    m = Add()([init, x])
    return m

def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1, weight_decay=1e-4, activation='relu'):
    """
    Creates a Wide Residual Network (WideResNet) with specified parameters.

    Args:
        input_dim: Dimension of the input data.
        nb_classes: Number of output classes for the network.
        N: Number of blocks in each group.
        k: Width factor for scaling the number of filters.
        dropout: Dropout rate.
        verbose: Verbosity mode.
        weight_decay: L2 regularization factor.
        activation: Activation function to use.

    Returns:
        A WideResNet model.
    """
    ip = Input(shape=input_dim)
    x = initial_conv(ip, weight_decay, activation=activation)
    nb_conv = 4  # Initial convolutional layer

    # First group of blocks
    x = expand_conv(x, 16, k, weight_decay=weight_decay, activation=activation)
    nb_conv += 2

    for i in range(N - 1):
        x = conv_block(x, k, 16, dropout, weight_decay, activation=activation)
        nb_conv += 2

    # Second group of blocks
    x = expand_conv(x, 32, k, strides=(2, 2), weight_decay=weight_decay, activation=activation)
    nb_conv += 2

    for i in range(N - 1):
        x = conv_block(x, k, 32, dropout, weight_decay, activation=activation)
        nb_conv += 2

    # Third group of blocks
    x = expand_conv(x, 64, k, strides=(2, 2), weight_decay=weight_decay, activation=activation)
    nb_conv += 2

    for i in range(N - 1):
        x = conv_block(x, k, 64, dropout, weight_decay, activation=activation)
        nb_conv += 2

    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation(activation)(x)

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    x = Dense(nb_classes, kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)
    model = Model(ip, x)

    if verbose: 
        print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model

