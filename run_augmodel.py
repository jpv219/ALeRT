##########################################################################
#### Augmented trained model run and evaluation
#### Author : Juan Pablo Valdes and Fuyue Liang
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from paths import PathConfig
import pandas as pd
from model_lib import ModelConfig

def main():

    # Load data to process
    path = PathConfig()

    case = input('Select a study to process raw datasets (sp_(sv)geom, (sv)surf, (sv)geom): ')
    label_package = []
    data_packs = []

    # Read package names generated at input.py to later import
    with open(os.path.join(path.input_savepath,case,'Load_Labels.txt'), 'r') as file:
        lines = file.readlines()

        for line in lines:
            label_package.append(line.split('\n')[0])

    # Save only train, test and random split packs
    label_package = [item for item in label_package if item not in ['full', 'PCA_info']]
    
    # Load pickle files
    for label in label_package:

        data_path = os.path.join(path.input_savepath,case,f'{label}.pkl')

        if os.path.exists(data_path):

            data_pack = pd.read_pickle(data_path)          
            data_packs.append(data_pack)

    # Augment training data with loaded sampled data
    X_train_aug = pd.concat([data_packs[0],data_packs[4]], ignore_index= True)
    y_train_aug = pd.concat([data_packs[1],data_packs[5]], ignore_index= True)

    model_choice = input('Select trained model to load and deploy: (dt, xgb, rf, svm, knn, mlp_br, mlp): ')

    # Model configurer
    m_config = ModelConfig()
    
    # selecting corresponding wrapper, hyperparams and model_name
    wrapper_model = m_config.get_wrapper(model_choice)
    model_params = m_config.get_hyperparameters(model_choice)
    model_name = m_config.get_model_name(model_choice)

    if model_choice in ['mlp', 'mlp_br']:
        is_mlp = True
        best_model_path = os.path.join(path.bestmodel_savepath, model_name,'best_model.keras')

        model_params['input_size'] = data_packs[0].shape[-1]
        model_params['output_size'] = data_packs[1].shape[-1]

        # Count the number of individual features in the input data
        features = data_packs[1].columns
        unique_features = set()

        for feat in features:
            # split the column number from the feature name
            name = feat.rsplit('_',1)[0]
            unique_features.add(name)

        n_features = len(unique_features)
        model_params['n_features'] = n_features

    else:
        is_mlp = False
        best_model_path = os.path.join(path.bestmodel_savepath, model_name,'best_model.pkl')

    # Instantiating the wrapper with the corresponding hyperparams
    model_instance = wrapper_model(**model_params)

    # Load best trained model
    best_model = model_instance.load_model(best_model_path, is_mlp)

    # Train model with new data, w/o further tuning or cross validation
    cv_options = {'do_kfold': False,
          'ksens' : False,
          'do_hp_tune': False}
    
    re_trained_model = model_instance.model_train([X_train_aug,y_train_aug], best_model,
                               cv_options, model_name)
    
    # Calling model evaluate with tuned model
    model_instance.model_evaluate(re_trained_model, [X_train_aug,y_train_aug,data_packs[2],data_packs[3]],
                                  case,pca=False)
    

if __name__ == "__main__":
    main()