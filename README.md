# Neural Network Compression Using Knowledge Distillation, Pruning, and Genetic Algorithm

## Overview

This final-year project proposes a novel method for reducing the size of neural networks. The approach combines knowledge transfer with genetic algorithms to identify essential components within the main neural networks, in line with the Lottery Ticket Hypothesis by Frank et al.

## Steps

1. **CNN Architecture Selection**: 
   - Chose various CNN architectures, including ResNet, GoogleNet and a random simple CNN architecture for the study.

2. **Transfer Learning**: 
   - Adapted CNN models to different datasets (CIFAR-10, CIFAR-100, MNIST) using transfer learning techniques.

3. **Pruning Techniques**: 
   - Applied pruning methods to reduce the size of neural networks while preserving their performance. This generated a population of pruned subnets to serve as the initial population for a genetic algorithm.

4. **Genetic Algorithm**: 
   - Utilized the initial population of pruned subnets to generate new solutions through the crossing operation, swapping different model layers to explore new solutions.

5. **Knowledge Transfer**: 
   - Utilized knowledge transfer with the initial model acting as the parent to guide the genetic algorithm in finding the optimal subnet.

## File Structure

- **Notebooks**: Each model architecture with each dataset has its own notebook, accessible and executable in Google Colab.
- **Python Files**:
  - `AG.py`: Genetic algorithm implementation.
  - `Dataset_loader.py`: Data loading utilities.
  - `TL.py`: Transfer learning functions.
  - `global_param.py`: Global parameters.
  - `my_lib.py`: Additional helper functions and classes.

## Getting Started

1. Clone the repository:
   ```sh
   git clone https://github.com/AmiraBelkis/Neural-Network-Reduction-via-Knowledge-Transfer.git

2. Open the notebook files in [Google Colab](https://colab.research.google.com/):
    - Navigate to the project directory in Colab.
    - Import the files in the .code folder to your Colab running session storage.

3. Run the notebooks directly in Colab.
