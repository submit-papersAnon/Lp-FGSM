# $l^p$-FGSM
An $l^p$ Norm Solution to Catastrophic Overfitting in Fast Adversarial Training

## Abstract
Adversarial training, recognized for enhancing the robustness of deep neural networks, faces the challenge of computational cost. Fast adversarial training methodologies, like the Fast Gradient Sign Method (FGSM), are efficient but susceptible to "catastrophic overfitting", leading to models robust against single-step attacks but vulnerable to multi-step variants. This work introduces an $l^p$-norm-based adversarial training framework to address this issue, providing an efficient solution to $l^\infty$ adversarial robustness and effectively mitigating Catastrophic Overfitting.

## Introduction
This repository contains the implementation of our novel $l^p$-norm-based adversarial training framework. Our approach primarily focuses on overcoming the limitations of existing fast adversarial training methods. By exploring the transition from $l^2$ to $l^\infty$ attacks, we introduce $l^p$-FGSM for creating fast single-step $l^p$ adversarial perturbations.
The code is in TensorFlow.
## Installation
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage
To use the code in this repository, follow these steps:

1. **Training a Model**: Run `python train_model.py` with the desired parameters. Example:

   ```bash
   python train_model.py --dataset CIFAR10 --epochs 30 --eps 8.0 --vareps 1e-12 --p 32.0 
   ```

2. **Evaluating Robustness**: Use `pgd_attack.py` to evaluate the robustness of your trained models against PGD attacks.

3. **Custom Training**: Modify `train_model.py` for custom adversarial training routines.

## Results
Our method demonstrates significant improvements in mitigating catastrophic overfitting. Here is a summary of the main results:
<p align="center">
  <img src="results_summary.png" alt="Results Summary" />
</p>

Performance on SVHN Dataset as a Function of $\epsilon$

| $255\cdot\epsilon $ | $l^p$-FGSM | RS-FGSM | N-FGSM | Grad Al | ZeroG |
|---|---|---|---|---|---|
| 2 | 94.4 ±0.44<br>**86.85 ±0.26** | **96.16 ±0.13**<br>86.17 ±0.17 | 96.04 ±0.24<br>86.46 ±0.12 | 96.01 ±0.25<br>86.44 ±0.15 | 96.08 ±0.22<br>86.47 ±0.17 |
| 4 | 94.36 ±0.68<br>**77.77 ±0.82** | **95.07 ±0.08**<br>71.25 ±0.43 | 94.56 ±0.18<br>72.54 ±0.21 | 94.57 ±0.24<br>72.18 ±0.22 | 94.83 ±0.19<br>71.64 ±0.24 |
| 6 | 92.77 ±0.69<br>**64.42 ±1.7** | **95.16 ±0.48**<br>0.0 ±0.0 | 92.27 ±0.36<br>58.44 ±0.18 | 92.55 ±0.26<br>57.36 ±0.27 | 93.52 ±0.24<br>51.77 ±0.58 |
| 8 | 91.14 ±0.69<br>**56.12 ±0.72** | **94.48 ±0.18**<br>0.0 ±0.0 | 89.59 ±0.48<br>45.64 ±0.21 | 90.16 ±0.36<br>43.88 ±0.16 | 92.43 ±1.33<br>35.96 ±2.78 |
| 10 | 90.95 ±1.2<br>**45.58 ±1.24** | **93.82 ±0.28**<br>0.0 ±0.0 | 86.78 ±0.88<br>33.98 ±0.48 | 87.26 ±0.73<br>32.88 ±0.36 | 90.36 ±0.33<br>21.36 ±0.37 |
| 12 | 89.06 ±0.36<br>**36.88 ±1.40** | **92.72 ±0.56**<br>0.0 ±0.0 | 81.49 ±1.66<br>26.17 ±0.88 | 84.12 ±0.44<br>23.64 ±0.42 | 88.11 ±0.47<br>14.16 ±0.38 |


Performance on CIFAR-10 Dataset as a Function of $\epsilon$

| $255\cdot\epsilon $ | $l^p$-FGSM | RS-FGSM | N-FGSM | Grad Al | ZeroG |
|---|---|---|---|---|---|
| 2 | 91.08 ±0.6<br>80.80 ±0.2 | **92.86 ±0.14**<br>**80.91 ±0.14** | 92.49 ±0.14<br>81.42 ±0.34 | 92.54 ±0.13<br>81.32 ±0.43 | 92.62 ±0.16<br>81.41 ±0.32 |
| 4 | 88.15 ±0.37<br>**69.53 ±0.8** | **90.74 ±0.23**<br>68.24 ±0.19 | 89.64 ±0.23<br>69.10 ±0.27 | 89.93 ±0.34<br>69.80 ±0.48 | 90.21 ±0.22<br>69.21 ±0.21 |
| 6 | 85.58 ±0.52<br>**59.18 ±0.52** | **88.25 ±0.22**<br>57.24 ±0.19 | 85.74 ±0.32<br>58.26 ±0.18 | 86.94 ±0.16<br>59.14 ±0.16 | 86.11 ±0.45<br>58.44 ±0.19 |
| 8 | 81.73 ±0.62<br>**51.33 ±0.63** | 83.61 ±1.77<br>0.0 ±0.0 | 81.64 ±0.35<br>49.51 ±0.27 | 82.16 ±0.21<br>50.12 ±0.17 | **84.16 ±0.21**<br>48.32 ±0.21 |
| 10 | 76.56 ±0.65<br>**45.96 ±0.71** | **82.17 ±1.48**<br>0.0 ±0.0 | 76.94 ±0.12<br>42.39 ±0.39 | 79.42 ±0.28<br>41.42 ±0.52 | 81.29 ±0.73<br>36.18 ±0.19 |
| 12 | 73.34 ±0.6<br>**41.18 ±1.46** | 78.64 ±0.74<br>0.0 ±0.0 | 72.18 ±0.17<br>36.82 ±0.27 | 73.72 ±0.82<br>35.16 ±0.77 | **79.33 ±0.92**<br>28.26 ±1.81 |
| 14 | 66.47 ±0.68<br>**38.72 ±0.84** | 73.27 ±2.84<br>0.0 ±0.0 | 67.86 ±0.46<br>31.68 ±0.68 | 66.41 ±0.52<br>30.85 ±0.34 | **78.18 ±0.66**<br>18.56 ±0.35 |
| 16 | 63.8 ±0.72<br>**37.14 ±1.04** | 68.68 ±2.43<br>0.0 ±0.0 | 56.75 ±0.44<br>25.11 ±0.43 | 57.88 ±0.74<br>26.24 ±0.43 | **75.43 ±0.89**<br>14.66 ±0.22 |



Performance on CIFAR-100 Dataset as a Function of $\epsilon$

| $255\cdot\epsilon $ | $l^p$-FGSM | RS-FGSM | N-FGSM | Grad Al | ZeroG |
|---|---|---|---|---|---|
| 2 | 66.83 ±0.12<br>**55.96 ±0.68** | **72.62 ±0.24**<br>51.62 ±0.56 | 71.52 ±0.14<br>52.24 ±0.35 | 71.61 ±0.23<br>51.51 ±0.48 | 71.64 ±0.22<br>52.63 ±0.64 |
| 4 | 61.36 ±0.37<br>**45.83 ±0.48** | **68.27 ±0.21**<br>39.56 ±0.14 | 66.51 ±0.48<br>39.96 ±0.31 | 67.09 ±0.19<br>39.81 ±0.48 | 67.21 ±0.18<br>39.61 ±0.32 |
| 6 | 59.08 ±0.52<br>**38.21 ±0.50** | **65.62 ±0.66**<br>26.61 ±2.79 | 61.42 ±0.63<br>30.99 ±0.27 | 62.86 ±0.1<br>32.11 ±0.24 | 63.65 ±0.12<br>30.28 ±0.51 |
| 8 | 53.54 ±0.64<br>**32.03 ±1.26** | 54.28 ±5.92<br>0.0 ±0.0 | 56.42 ±0.65<br>26.71 ±0.68 | 58.55 ±0.41<br>26.97 ±0.61 | **60.78 ±0.24**<br>23.72 ±0.16 |
| 10 | 50.06 ±0.48<br>**27.28 ±0.87** | 46.18 ±4.88<br>0.0 ±0.0 | 51.51 ±0.61<br>23.11 ±0.49 | 53.85 ±0.73<br>22.64 ±0.61 | **61.11 ±0.39**<br>15.15 ±0.45 |
| 12 | 47.17 ±0.24<br>**24.51 ±0.63** | 35.86 ±0.27<br>0.0 ±0.0 | 46.42 ±0.56<br>19.32 ±0.51 | 46.94 ±0.86<br>19.94 ±0.65 | **58.36 ±0.15**<br>11.12 ±0.66 |
| 14 | 43.26 ±0.28<br>**22.27 ±1.02** | 24.42 ±1.38<br>0.0 ±0.0 | 42.14 ±0.36<br>16.62 ±0.44 | 42.63 ±0.5<br>16.96 ±0.14 | **56.24 ±0.16**<br>8.81 ±0.34 |
| 16 | 40.62 ±1.7<br>**18.23 ±1.53** | 21.47 ±5.21<br>0.0 ±0.0 | 38.37 ±0.48<br>14.29 ±0.38 | 36.17 ±0.45<br>14.23 ±0.26 | **56.42 ±0.29**<br>4.92 ±0.38 |




## Contributing
Contributions to this project are welcome. Please submit a pull request or open an issue for features or bug fixes.

## This repository is based on:
https://github.com/tml-epfl/understanding-fast-adv-training 

https://github.com/rohban-lab/catastrophic_overfitting

https://github.com/pdejorge/N-FGSM 

https://github.com/mahyarnajibi/FreeAdversarialTraining 



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use our method or this codebase in your research, please cite our work:

```bibtex
@article{TBA,
  title={An $l^p$ Norm Solution to Catastrophic Overfitting in Fast Adversarial Training},
  author={TBA},
  journal={Conference/Journal Name},
  year={2023}
}
```
