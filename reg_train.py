##########################################################################
#### Regression model train and deployment
#### Author : Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from paths import PathConfig
from model_lib import ModelConfig
from data_utils import DataLoader

####

class SetModel(PathConfig):

    def __init__(self, case, model_choice):

        super().__init__()

        # Mixer study deployed
        self.case = case
        self.model_choice = model_choice
        self.inidata_dir = os.path.join(self.input_savepath, case, 'ini')

        # Model configurer
        m_config = ModelConfig()
        
        self._wrapper_model = m_config.get_wrapper(model_choice)
        self._model_params = m_config.get_hyperparameters(model_choice)
        self._model_name = m_config.get_model_name(model_choice)

        # Instantiating the wrapper with the corresponding hyperparams
        self._model_instance = self._wrapper_model(**self._model_params)

    @property
    def model_name(self):
        return self._model_name
        
    @property
    def wrapper_model(self):
        return self._wrapper_model
    
    @property
    def model_params(self):
        return self._model_params
    
    def update_model_params(self,key,value):
        self._model_params[key] = value
    
    @property
    def model_instance(self):
        return self._model_instance
    
    @model_instance.setter
    def model_instance(self,params):
        self._model_instance = self.wrapper_model(**params)
    
    def load_data(self):
            
        # Dataloader
        dataloader = DataLoader(self.case)

        # load initial data packs and retrieve whether pca has been executed
        data_packs = dataloader.load_packs(self.inidata_dir)

        # Verify if PCA has been used on input data pre-processing, after loading packs!
        pca = dataloader.pca

        return data_packs, pca

    def traintune_config(self):

        # Add input_size, output_size and n_features for MLP models and negate early kfold
        if self.model_choice in ['mlp', 'mlp_br']:
            is_mlp = True

            do_kfold = 'n'
            ksens = 'n'

        # only ask for early kfold on sklearn native models
        else:
            is_mlp = False

            do_kfold = input('Perform pre-Kfold cross validation? (y/n): ')
            
            # Decide whether to do pre-kfold and include k sensitivity
            if do_kfold.lower() == 'y':
                ksens = input('Include K-sensitivity? (y/n): ')
            else:
                ksens = 'n'

        do_hp_tune = input('Perform hyperparameter tuning cross-validation? (y/n): ')

        cv_options = {'is_mlp': is_mlp,
            'do_kfold': True if do_kfold.lower() == 'y' else False,
            'ksens' : True if ksens.lower() == 'y' else False,
            'do_hp_tune': True if do_hp_tune.lower() == 'y' else False}

        return cv_options

class TrainModel(PathConfig):

    def __init__(self, model_setter: SetModel, **kwargs) -> None:
        
        super().__init__()
        # Composing SetModel class within TrainModel to access model configuration: wrapper, params and instances
        self.setter = model_setter

        self.cv_options = kwargs
        self.is_mlp = self.cv_options.get('is_mlp',None)
    
    def train_model(self, data_packs):

        # Add input_size, output_size and n_features for MLP models
        if self.is_mlp:
            self.setter.update_model_params('input_size',data_packs[0].shape[-1])
            self.setter.update_model_params('output_size',data_packs[1].shape[-1])

            # Count the number of individual features in the input data
            features = data_packs[1].columns
            unique_features = set()

            for feat in features:
                # split the column number from the feature name
                name = feat.rsplit('_',1)[0]
                unique_features.add(name)

            n_features = len(unique_features)
            self.setter.update_model_params('n_features',n_features)
        
            # Re-instantiate the wrapper with the updated hyperparams in case of MLP
            self.setter.model_instance = self.setter.model_params

        # Getting regressor object from wrapper
        model = self.setter.model_instance.init_model()
        
        # Regression training and evaluation
        tuned_model = self.setter.model_instance.model_train(data_packs, model, 
                                            self.cv_options, self.setter.model_name)
        
        return tuned_model

    def save_best_model(self,tuned_model):

        # Save best perfoming trained model based on model_train cross_validation filters and steps selected by the user
        best_model_path = os.path.join(self.bestmodel_savepath, self.setter.model_name)

        # Check if the subfolder for the model exists
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        self.setter.model_instance.save_model(tuned_model,best_model_path,self.is_mlp)
    
def main():

    case = input('Select a study from where to load proccessed data packs (sp_(sv)geom): ')

    # Model selection from user input
    model_choice = input('Select a regressor to train and deploy (dt, xgb, rf, svm, knn, mlp_br, mlp): ')

    setter = SetModel(case, model_choice)

    data_packs, pca = setter.load_data()
    cv_options = setter.traintune_config()

    # Model trainer
    trainer = TrainModel(setter, **cv_options)

    tuned_model = trainer.train_model(data_packs)

    print('-'*72)
    print('-'*72)
    print(f'Saving {setter.model_name} best model...')

    trainer.save_best_model(tuned_model)
    
    # Calling model evaluate with tuned model
    setter.model_instance.model_evaluate(tuned_model, setter.model_name, data_packs,case,pca, datasample = 'ini')

if __name__ == "__main__":
    main()