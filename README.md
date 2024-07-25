# ALeRT: Active Learning Regression Toolbox

Welcome to the ALeRT (Active Learning Regression Toolbox) repository! This project provides a flexible framework for training and executing different machine learning regression models on hydrodynamic features of mixing devices. ALeRT allows for hyperparameter tuning, k-fold cross-validation, and active learning techniques to generate new data points for augmented model training.

## Table of Contents
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Workflow](#workflow)
- [Modules Overview](#modules-overview)
- [License](#license)
- [Contact](#contact)

## Project Structure

Here's an overview of the file and folder structure in this repository:

```plaintext
.
├── base_model.py
├── best_models
│   ├── Decision_Tree
│   ├── K_Nearest_Neighbours
│   ├── MLP_Branched_Network
│   ├── Multi_Layer_Perceptron
│   ├── Random_Forest
│   ├── Support_Vector_Machine
│   └── XGBoost
├── config
│   └── config_paths.ini
├── csv_data
│   └── sp_geom
│       ├── dt
│       ├── ini
├── data_utils.py
├── DOE
│   └── sp_geom
│       ├── dt
│       ├── ini
├── figs
│   └── sp_geom
│       ├── dt
│       ├── ini
│       └── random
├── input_data
│   └── sp_geom
│       └── ini
├── input.py
├── model_lib.py
├── models
│   ├── Decision_Tree
│   │   └── hyperparam_tune
│   ├── K_Nearest_Neighbours
│   │   └── hyperparam_tune
│   ├── MLP_Branched_Network
│   │   └── hyperparam_tune
│   ├── Multi_Layer_Perceptron
│   │   └── hyperparam_tune
│   ├── Random_Forest
│   │   └── hyperparam_tune
│   ├── Support_Vector_Machine
│   │   └── hyperparam_tune
│   └── XGBoost
│       └── hyperparam_tune
├── model_utils.py
├── paths.py
├── pca_models
│   └── sp_geom
│       ├── dt
│       ├── ini
│       └── random
├── reg_train.py
├── resample
│   └── sp_geom
│       ├── dt
│       │   ├── log_rules
│       │   │   ├── dt_rules_1.log
│       │   │   ├── dt_rules_2.log
│       ├── gsx
│       │   └── log_rules
│       │       └── gsx_rules.log
│       └── random
├── run_augmodel.py
└── sampling.py
```

## Usage

To use this repository, follow these steps:

1. **Install Dependencies**: Ensure you have all the necessary Python packages installed. If there is a `requirements.txt` file, use `pip install -r requirements.txt`.

2. **Generate Features and Targets**: Run `input.py` to process the csv data and labels from the DOE files to generate the features and targets and testing sets for the regression. A selection of which targets to include can be done at this stage, as well as carrying out dimensionality reduction via Principal Component Analysis (PCA)

3. **Train and Evaluate Models**: Run `reg_train.py` to choose an available regression model to be trained and evaluated, with the option to include hyperparameter tuning and kfold cross-validation.

4. **Generate New Data Points**: Use `sampling.py` to apply active learning techniques (uncertainty exploration from Decision tree or Greedy Sampling on the inputs) and generate rules for the new data points that should be added to the existing database.

5. **Re-train with Augmented Data**: Execute `run_augmodel.py` to re-train the regression models with the new data points obtained.

## Workflow

The typical workflow for ALeRT involves the following steps:

1. **Initial Data Preparation**: Run `input.py` to set up the initial data for training.

2. **Model Training**: Use `reg_train.py` to train and evaluate models. This script also supports hyperparameter tuning and k-fold cross-validation.

3. **Active Learning**: Execute `sampling.py` to identify new data points to obtain through active learning techniques.

4. **Model Re-training**: Use `run_augmodel.py` to re-train the models with the newly acquired data points.

## Modules Overview

- `base_model.py`: Contains the backbone for all regression models (abstract parent classes).
- `data_utils.py`: Provides utilities for data management, such as loading, pre-processing, scaling, saving, and augmenting, as well as carrying out PCA dimensionality reduction.
- `model_lib.py`: Uses `base_model.py` to build and manage different regression models. Models available are : Decision Tree, XGBoost, Random Forest, K-Nearest-Neighbours, Support Vector Machine, Multi-Layer Perceptron and custom built MLP-branched
- `model_utils.py`: Contains utility functions for model cross-validation, hyperparameter tuning and evaluation.
- `input.py`: Script to generate train and test pickle files for later scripts to interpret as features and targets.
- `reg_train.py`: Script to train and evaluate the regression models.
- `sampling.py`: Script for active learning and generation of new data points.
- `run_augmodel.py`: Script to re-train regression models with augmented data.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

## Contact

If you have any questions or suggestions, feel free to open an issue or contact the repository owner at [j.valdes20@imperial.ac.uk](mailto:j.valdes20@imperial.ac.uk).

