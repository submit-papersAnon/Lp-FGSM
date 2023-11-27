import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress most TensorFlow logs

import tensorflow as tf
import argparse
import numpy as np
from tensorflow.keras.optimizers import Adam, AdamW, SGD

# Import necessary modules and functions
from architectures import create_wide_residual_network, PreActResNet18
from checkpoints import AdversarialCheckpoint, AdversarialCheckpoint_Epochs
from cyclic_lr import CyclicLR
from lp_fgsm import Lp_FGSM
from dataloader import load_and_preprocess_dataset

def train_model(args):
    seed = 88888888+args.seed
    np.random.seed(seed)  # For NumPy random numbers
    tf.random.set_seed(seed)  # For TensorFlow random numbers

    # Variants of cross entropy loss
    ccs = tf.keras.losses.CategoricalCrossentropy(reduction='sum') # reduction sum
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) # reduction none to get access to each loss
    
    datagen, (x_train, y_train), (x_test, y_test), num_classes = load_and_preprocess_dataset(args.dataset_name)

    # Model selection based on dataset
    if args.dataset_name == 'SVHN':
        model = PreActResNet18(input_shape=(32, 32, 3), num_classes=num_classes, weight_decay=args.weight_decay, dropout_rate=args.dropout, activation='relu')
    elif args.dataset_name in ['CIFAR10', 'CIFAR100']:
        model = create_wide_residual_network((32, 32, 3), nb_classes=num_classes, N=4, k=8, dropout=args.dropout, weight_decay=args.weight_decay, activation='relu')
    else:
        raise ValueError("Unknown dataset")

    model.compile(optimizer=Adam(learning_rate=args.learning_rate, weight_decay=args.weight_decay), loss="categorical_crossentropy", metrics=["accuracy"])
    
    if args.pretrain_epochs > 0:
        model.fit(datagen.flow(x_train, y_train, batch_size=args.batch_size), validation_data=(x_test, y_test), epochs=args.pretrain_epochs)

    model_lp = Lp_FGSM(base_model=model, p=args.p, eps=args.eps, vareps=args.vareps, add_noise=args.add_noise)

    if args.cyclic_lr:
        lr_max, lr_min = 0.2, 0.001
        cyclic_lr_schedule = CyclicLR(lr_max, lr_min, args.epochs)
        model_lp.compile(optimizer=SGD(learning_rate=cyclic_lr_schedule, weight_decay=args.weight_decay, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])
    else:
        model_lp.compile(optimizer=Adam(learning_rate=args.learning_rate, weight_decay=args.weight_decay), loss="categorical_crossentropy", metrics=["accuracy"])

    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    #adv_callback = AdversarialCheckpoint_Epochs(dataset=dataset_test, epsilon=args.eps, attack_iters=50, restarts=1, adv_batch_size=1024)
    adv_callback = AdversarialCheckpoint(dataset=dataset_test, epsilon=args.eps, attack_iters=50, restarts=1, adv_batch_size=1024)
    callbacks_list = [adv_callback]

    model_lp.fit(datagen.flow(x_train, y_train, batch_size=args.batch_size), epochs=args.epochs, validation_data=(x_test, y_test), callbacks=callbacks_list, verbose=1)

    adv_accuracy_linf = adv_callback.adv_accuracy
    test_accuracy = adv_callback.test_accuracy

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Log results in the logs folder
    filename = f"logs/{args.dataset_name}_p{args.p}_eps{args.eps}_vareps{args.vareps}_addnoise{args.add_noise}_seed{args.seed}.txt"
    with open(filename, "w") as file:
        file.write(f"Adversarial Accuracy (Linf): {adv_accuracy_linf * 100:.2f}%\n")
        file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")

    return adv_accuracy_linf, test_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network model.')

    # Define the arguments
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (e.g., SVHN, CIFAR10, CIFAR100)')
    parser.add_argument('--eps', type=float, default=8.0, help='Perturbation limit for adversarial training')
    parser.add_argument('--vareps', type=float, default=1e-12, help='Small constant for numerical stability')
    parser.add_argument('--p', type=float, default=32.0, help='Norm degree for Lp norm')

    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for L2 regularization')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the model')
    parser.add_argument('--pretrain_epochs', type=int, default=0, help='Number of pretraining epochs')

    parser.add_argument('--seed', type=int, default=0, help='Random seed number')

    parser.add_argument('--add_noise', action='store_true', help='Add noise during training')
    parser.add_argument('--no_add_noise', action='store_false', help='Do not add noise during training')
    parser.set_defaults(add_noise=True)

    parser.add_argument('--cyclic_lr', action='store_true', help='Use cyclic learning rate')
    parser.add_argument('--no_cyclic_lr', action='store_false', help='Do not use cyclic learning rate')
    parser.set_defaults(cyclic_lr=False)

    args = parser.parse_args()
    train_model(args)
