#!/bin/bash

# Suppress most TensorFlow logs
export TF_CPP_MIN_LOG_LEVEL=2

###########################
# Code to create Figure 6
# This section runs training for various datasets, seeds, and p values.
###########################

# Array of dataset names
datasets=("SVHN" "CIFAR10" "CIFAR100")

# Array of p values
p_values=(2 4 8 16 32 64 128 256)

# Loop over datasets
for dataset in "${datasets[@]}"
do
    # Loop over seeds
    for seed in {0..4}
    do
        # Loop over p values
        for p in "${p_values[@]}"
        do
            echo "Running with dataset=$dataset, seed=$seed, and p=$p"
            python train_model.py --dataset_name $dataset --epochs 30 --eps 8.0 --p $p --no_add_noise --learning_rate 0.001 --weight_decay 0.0 --batch_size 1024 --dropout 0.0 --pretrain_epochs 0 --no_cyclic_lr --seed $seed
        done
    done
done

###########################
# Code to create Figure 7
# This section runs training for SVHN, CIFAR10, and CIFAR100 datasets with varying epsilon values.
###########################

# Define epsilon values
eps_values=(2 4 6 8 10 12 14 16)
eps_values_svhn=(2 4 6 8 10 12)

# SVHN Dataset
for seed in {0..4}
do
    for eps in "${eps_values_svhn[@]}"
    do
        echo "Running SVHN with seed=$seed, epsilon=$eps"
        python train_model.py --dataset_name SVHN --epochs 30 --eps $eps --p 32.0 --add_noise --learning_rate 0.001 --weight_decay 5e-4 --batch_size 1024 --dropout 0.1 --pretrain_epochs 0 --cyclic_lr --seed $seed
    done
done

# CIFAR10 Dataset
for seed in {0..4}
do
    for eps in "${eps_values[@]}"
    do
        echo "Running CIFAR10 with seed=$seed, epsilon=$eps"
        python train_model.py --dataset_name CIFAR10 --epochs 30 --eps $eps --p 32.0 --add_noise --learning_rate 0.001 --weight_decay 5e-4 --batch_size 1024 --dropout 0.1 --pretrain_epochs 0 --cyclic_lr --seed $seed
    done
done

# CIFAR100 Dataset
for seed in {0..4}
do
    for eps in "${eps_values[@]}"
    do
        echo "Running CIFAR100 with seed=$seed, epsilon=$eps"
        python train_model.py --dataset_name CIFAR100 --epochs 30 --eps $eps --p 16.0 --add_noise --learning_rate 0.001 --weight_decay 5e-4 --batch_size 1024 --dropout 0.0 --pretrain_epochs 0 --cyclic_lr False --seed $seed
    done
done

