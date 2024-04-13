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

# Global paths configuration
PATH = PathConfig()
####

def check_pca(label_package):
    # Checking in PCA has been applied to the dataset
    if 'PCA_info' in label_package:
        pca = True
    else:
        pca = False
    
    return pca

def load_packs(dir):
    
    label_package = []
    data_packs = []

    labelfile_dir = os.path.join(dir,'Load_Labels.txt')
    
    # Read package names generated at input.py to later import
    with open(labelfile_dir, 'r') as file:
        lines = file.readlines()

        for line in lines:
            label_package.append(line.split('\n')[0])

    pca = check_pca(label_package)
    
    # Save only train, test packs
    label_package = [item for item in label_package if item not in ['full', 'PCA_info']]
    
    # Load pickle files
    for label in label_package:

        data_path = os.path.join(dir,f'{label}.pkl')

        if os.path.exists(data_path):

            data_pack = pd.read_pickle(data_path)          
            data_packs.append(data_pack)
    
    return data_packs, pca
        
def augment_data(case, data_sample : str):

    data_aug_dir = os.path.join(PATH.resample_savepath, case, data_sample)
    ini_data_dir = os.path.join(PATH.input_savepath, case, 'ini')

    aug_packs,_ = load_packs(data_aug_dir)
    ini_packs,pca = load_packs(ini_data_dir)

    # Augment training data with loaded sampled data
    X_train_aug = pd.concat([ini_packs[0],aug_packs[0]], ignore_index= True)
    y_train_aug = pd.concat([ini_packs[1],aug_packs[1]], ignore_index= True)

    # Returning the newly augmented training sets and the original testing sets split in input.py
    data_packs = [X_train_aug, y_train_aug,ini_packs[2],ini_packs[3]]

    return data_packs, pca

def main():

    case = input('Select a study to process raw datasets (sp_(sv)geom, (sv)surf, (sv)geom): ')

    model_choice = input('Select a trained model to load and deploy (dt, xgb, rf, svm, knn, mlp_br, mlp): ')

    data_choice = input('Select the dataset sample to augment (random, dt, gsx): ')

    # augment dataset with selected sample data
    data_packs,pca = augment_data(case, data_choice)
    
    # Model configurer
    m_config = ModelConfig()
    
    # selecting corresponding wrapper, hyperparams and model_name
    wrapper_model = m_config.get_wrapper(model_choice)
    model_params = m_config.get_hyperparameters(model_choice)
    model_name = m_config.get_model_name(model_choice)

    if model_choice in ['mlp', 'mlp_br']:
        is_mlp = True
        best_model_path = os.path.join(PATH.bestmodel_savepath, model_name,'best_model.keras')

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
        best_model_path = os.path.join(PATH.bestmodel_savepath, model_name,'best_model.pkl')

    # Instantiating the wrapper with the corresponding hyperparams
    model_instance = wrapper_model(**model_params)

    # Load best trained model
    best_model = model_instance.load_model(best_model_path, is_mlp)

    # Train model with new data, w/o further tuning or cross validation
    cv_options = {'do_kfold': False,
          'ksens' : False,
          'do_hp_tune': False}
    
    re_trained_model = model_instance.model_train(data_packs, best_model,
                               cv_options, model_name)
    
    # Calling model evaluate with tuned model
    model_instance.model_evaluate(re_trained_model, data_packs,
                                  case,pca, data_choice)
    

if __name__ == "__main__":
    main()