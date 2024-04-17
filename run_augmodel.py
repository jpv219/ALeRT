##########################################################################
#### Augmented trained model run and evaluation
#### Author : Juan Pablo Valdes and Fuyue Liang
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from paths import PathConfig
from model_lib import ModelConfig
from data_utils import DataLoader

# Global paths configuration
PATH = PathConfig()
####
   
def main():

    case = input('Select a study from where to load proccessed data packs (sp(sv)_geom): ')

    model_choice = input('Select a trained model to load and deploy (dt, xgb, rf, svm, knn, mlp_br, mlp): ')

    data_choice = input('Select the dataset sample to augment with (random, dt, gsx): ')

    dataloader = DataLoader(case)
    
    # augment dataset with selected sample data
    data_packs = dataloader.augment_data(data_choice)

    print(f'Augmented training data now has size: {data_packs[0].shape[0]}')
    print('-'*72)

    pca = dataloader.pca
    
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