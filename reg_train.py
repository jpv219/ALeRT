##########################################################################
#### Regression model train and deployment
#### Author : Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import numpy as np
import pandas as pd
import configparser
import os
from abc import ABC, abstractmethod
#Model imports
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class PathConfig:

    def __init__(self):
        self._config = configparser.ConfigParser()
        self._config.read(os.path.join(os.getcwd(), 'config/config_paths.ini'))

    @property
    def fig_savepath(self):
        return self._config['Path']['figs']

    @property
    def input_savepath(self):
        return self._config['Path']['input']

    @property
    def raw_datapath(self):
        return self._config['Path']['csv']
    
    @property
    def label_datapath(self):
        return self._config['Path']['doe']

# REGRESSOR PARENT CLASS

class Regressor(ABC):
    """Abstract base class for regression models."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def init_model(self):
        """Initialize the regression model."""
        pass

    def model_eval(self, **kwargs):
        self.kwargs = kwargs

        # Reading data arrays for model fit and eval
        X_train_arr = self.kwargs.get('X_train',None).to_numpy()
        y_train_arr = self.kwargs.get('y_train',None).to_numpy()
        X_test_arr = self.kwargs.get('X_test').to_numpy()
        y_test_arr = self.kwargs.get('y_test').to_numpy()

        model = self.kwargs.get('model')

        # Fit model from wrapper
        model.fit(X_train_arr,y_train_arr)

        # Carry out predictions and evaluate model performance
        y_pred = model.predict(X_test_arr)
        r2 = r2_score(y_test_arr,y_pred)
        mae = mean_absolute_error(y_test_arr,y_pred)
        mse = mean_squared_error(y_test_arr,y_pred)

        return r2,mae,mse

# Individual regressor child classes
    
class DecisionTreeWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):

        max_depth = self.kwargs.get('max_depth',None)

        if max_depth is None:
            raise ValueError('Max_depth is required for Decision Tree Regressor')
        
        return DecisionTreeRegressor(max_depth=max_depth, random_state=self.kwargs.get('random_state',2024))
    
    def model_eval(self, **kwargs):
        return super().model_eval(**kwargs)

class XGBoostWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):

        max_depth = self.kwargs.get('max_depth',None)
        n_estimators = self.kwargs.get('n_estimators',None)

        if max_depth is None or n_estimators is None:
            raise ValueError('Missing input arguments: Max_depth/n_estimators')
        
        return XGBRegressor(max_depth=max_depth, n_estimators = n_estimators , random_state=self.kwargs.get('random_state',2024))
    
    def model_eval(self, **kwargs):
        return super().model_eval(**kwargs)
    
class RandomForestWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):

        n_estimators = self.kwargs.get('n_estimators',None)

        if n_estimators is None:
            raise ValueError('n_estimators is required for Random Forest Regressor')
        
        return RandomForestRegressor(n_estimators = n_estimators, random_state=self.kwargs.get('random_state',2024))
    
    def model_eval(self, **kwargs):
        return super().model_eval(**kwargs)
    
class SVMWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):
        
        c_coef = self.kwargs.get('C', None)
        epsilon = self.kwargs.get('epsilon',None)

        if c_coef is None or epsilon is None:
            raise ValueError(' C and epsilon required for SVM')

        return SVR(C=c_coef,epsilon=epsilon)
    
    def model_eval(self, **kwargs):
        return super().model_eval(**kwargs)

class KNNWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):
        
        n_neighbours = self.kwargs.get('n_neighbours', None)

        if n_neighbours is None:
            raise ValueError('n_neighbours required for KNN')
        
        return KNeighborsRegressor(n_neighbors=n_neighbours)
    
    def model_eval(self, **kwargs):
        return super().model_eval(**kwargs)


def main():

    # Load data to process
    path = PathConfig()

    case = input('Select case to load data for model train and deployment (sp_geom, geom, surf): ')
    label_package = []
    data_packs = []

    # Read package names to later import
    with open(os.path.join(path.input_savepath,case,'Load_Labels.txt'), 'r') as file:
        lines = file.readlines()

        for line in lines:
            label_package.append(line.split('\n')[0])

    # Save only train and test packs
    label_package =  [item for item in label_package if item not in ['full', 'PCA_res']]
    
    # Load pickle files
    for label in label_package:

        data_path = os.path.join(path.input_savepath,case,f'{label}.pkl')

        if os.path.exists(data_path):

            data_pack = pd.read_pickle(data_path)           
            data_packs.append(data_pack)

    X_train, y_train, X_test, y_test = data_packs[:4]
    
    ## Regressor instances, labels and hyperparameters
    
    model_labels = {'dt': 'Decision Tree', 'xgb': 'XGBoost', 
                    'rf': 'Random Forest',
                    'svm': 'Support Vector Machine',
                    'knn': 'K-Nearest Neighbours'}
    
    wrapper_dict = {'dt': DecisionTreeWrapper, 'xgb': XGBoostWrapper, 
                    'rf': RandomForestWrapper,
                    'svm': SVMWrapper,
                    'knn': KNNWrapper}
    
    hyperparameters = {'dt': {'max_depth': 5}, 'xgb': {'max_depth': 5, 'n_estimators': 100}, 
                    'rf': {'n_estimators': 100},
                    'svm': {'C': 1, 'epsilon': 0.1},
                    'knn': {'n_neighbours': 10}}
    
    # Model selection from user input
    model_choice = input('Select a regressor to train and deploy (dt, xgb, rf, svm, knn): ')

    # selecting corresponding wrapper
    wrapper_model = wrapper_dict.get(model_choice)

    model_params = hyperparameters.get(model_choice)

    model_label = model_labels.get(model_choice)

    # Instantiating the wrapper with the corresponding hyperparams
    model_instance = wrapper_model(**model_params)

    model = model_instance.init_model()

    model.fit(X_train,y_train)

    # Regression training and evaluation
    #r2, mae, mse = model_instance.model_eval(X_train = X_train, y_train = y_train, 
                                             #X_test = X_test, y_test = y_test, model=model)
 

if __name__ == "__main__":
    main()