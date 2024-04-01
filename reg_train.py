##########################################################################
#### Regression model train and deployment
#### Author : Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from base_model import PathConfig
from model_lib import ModelConfig

def main():

    # Load data to process
    path = PathConfig()

    case = input('Select a study to process raw datasets (sp_(sv)geom, (sv)surf, (sv)geom): ')
    label_package = []
    data_packs = []

    # Read package names to later import
    with open(os.path.join(path.input_savepath,case,'Load_Labels.txt'), 'r') as file:
        lines = file.readlines()

        for line in lines:
            label_package.append(line.split('\n')[0])

    # Checking in PCA has been applied to the dataset
    if 'PCA_info' in label_package:
        pca = True
    else:
        pca = False
    
    # Save only train and test packs
    label_package =  [item for item in label_package if item not in ['full', 'PCA_info']]
    
    # Load pickle files
    for label in label_package:

        data_path = os.path.join(path.input_savepath,case,f'{label}.pkl')

        if os.path.exists(data_path):

            data_pack = pd.read_pickle(data_path)          
            data_packs.append(data_pack)
    
    # Model selection from user input
    model_choice = input('Select a regressor to train and deploy (dt, xgb, rf, svm, knn, mlp_reg, mlp): ')
    
    # selecting corresponding wrapper, hyperparams and model_name
    wrapper_model = ModelConfig.get_wrapper(model_choice)

    model_params = ModelConfig.get_hyperparameters(model_choice)

    model_name = ModelConfig.get_model_name(model_choice)

    # Add input_size and output_size for MLP models and negate early kfold
    if model_choice in ['mlp', 'mlp_reg']:
        model_params['input_size'] = data_packs[0].shape[-1]
        model_params['output_size'] = data_packs[1].shape[-1]
        skip_kfold = 'y'
        ksens = 'n'
    
    # only ask for early kfold on sklearn native models
    else:

        skip_kfold = input('Skip pre-Kfold cross validation? (y/n): ')
    
        # Decide whether to do pre-kfold and include k sensitivity
        if skip_kfold.lower() == 'n':
            ksens = input('Include K-sensitivity? (y/n): ')
        else:
            ksens = 'n'
    
    skip_hp_tune = input('Skip hyperparameter tuning cross-validation? (y/n): ')

    cv_options = {'skip_kfold': True if skip_kfold.lower() == 'y' else False,
          'ksens' : True if ksens.lower() == 'y' else False,
          'hp_tune': True if skip_hp_tune.lower() == 'y' else False}

    # Instantiating the wrapper with the corresponding hyperparams
    model_instance = wrapper_model(**model_params)

    # Getting regressor object from wrapper
    model = model_instance.init_model()

    # Regression training and evaluation
    tuned_model = model_instance.model_train(data_packs, model, 
                                        cv_options, model_name)
    # Calling model evaluate with tuned model
    model_instance.model_evaluate(tuned_model, data_packs,case,pca)

if __name__ == "__main__":
    main()