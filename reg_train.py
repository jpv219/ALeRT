##########################################################################
#### Regression model train and deployment
#### Author : Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from paths import PathConfig
from model_lib import ModelConfig

def main():

    # Load data to process
    path = PathConfig()

    case = input('Select a study to process raw datasets (sp_(sv)geom, (sv)surf, (sv)geom): ')
    label_package = []
    data_packs = []

    # Read package names to later import
    with open(os.path.join(path.input_savepath,case,'ini','Load_Labels.txt'), 'r') as file:
        lines = file.readlines()

        for line in lines:
            label_package.append(line.split('\n')[0])

    # Checking in PCA has been applied to the dataset
    if 'PCA_info' in label_package:
        pca = True
    else:
        pca = False
    
    # Save only train and test packs: [X,y train, X,y test and X,y random]
    label_package =  [item for item in label_package if item not in ['full', 'PCA_info']]
    
    # Load pickle files
    for label in label_package:

        data_path = os.path.join(path.input_savepath,case,'ini',f'{label}.pkl')

        if os.path.exists(data_path):

            data_pack = pd.read_pickle(data_path)          
            data_packs.append(data_pack)
    
    # Model selection from user input
    model_choice = input('Select a regressor to train and deploy (dt, xgb, rf, svm, knn, mlp_br, mlp): ')
    
    # Model configurer
    m_config = ModelConfig()
    
    # selecting corresponding wrapper, hyperparams and model_name
    wrapper_model = m_config.get_wrapper(model_choice)

    model_params = m_config.get_hyperparameters(model_choice)

    model_name = m_config.get_model_name(model_choice)

    # Add input_size, output_size and n_features for MLP models and negate early kfold
    if model_choice in ['mlp', 'mlp_br']:
        model_params['input_size'] = data_packs[0].shape[-1]
        model_params['output_size'] = data_packs[1].shape[-1]
        is_mlp = True

        # Count the number of individual features in the input data
        features = data_packs[1].columns
        unique_features = set()

        for feat in features:
            # split the column number from the feature name
            name = feat.rsplit('_',1)[0]
            unique_features.add(name)

        n_features = len(unique_features)
        model_params['n_features'] = n_features

        do_kfold = 'n'
        ksens = 'n'
    
    # only ask for early kfold on sklearn native models
    else:

        do_kfold = input('Perform pre-Kfold cross validation? (y/n): ')
    
        # Decide whether to do pre-kfold and include k sensitivity
        if do_kfold.lower() == 'y':
            ksens = input('Include K-sensitivity? (y/n): ')
        else:
            ksens = 'n'

        is_mlp = False
    
    do_hp_tune = input('Perform hyperparameter tuning cross-validation? (y/n): ')

    cv_options = {'do_kfold': True if do_kfold.lower() == 'y' else False,
          'ksens' : True if ksens.lower() == 'y' else False,
          'do_hp_tune': True if do_hp_tune.lower() == 'y' else False}

    # Instantiating the wrapper with the corresponding hyperparams
    model_instance = wrapper_model(**model_params)

    # Getting regressor object from wrapper
    model = model_instance.init_model()

    # Regression training and evaluation
    tuned_model = model_instance.model_train(data_packs, model, 
                                        cv_options, model_name)
    
    print('-'*72)
    print('-'*72)
    print(f'Saving {model_name} best model...')
    
    # Save best perfoming trained model based on model_train cross_validation filters and steps selected by the user
    best_model_path = os.path.join(path.bestmodel_savepath, model_name)
    model_instance.save_model(tuned_model,best_model_path,is_mlp)

    # Calling model evaluate with tuned model
    model_instance.model_evaluate(tuned_model, data_packs,case,pca, datasample = 'ini')

if __name__ == "__main__":
    main()