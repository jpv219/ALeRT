##########################################################################
#### Model train and evaluate utilities
#### Author : Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import numpy as np
import configparser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import RepeatedKFold, StratifiedKFold, KFold, cross_validate

############################# PATH UTILITIES ##############################################
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
    
    @property
    def model_savepath(self):
        return self._config['Path']['models']
    


class EarlyStopping(PathConfig):

    def __init__(self):
        super().__init__()

class KFoldCrossValidator(PathConfig):

    def __init__(self):
        super().__init__()

    def gen_kfold_cv(self, X, y, model, label, min_k=3, kfolds=50):
        
        # number of kfolds to try as hyperparameter for the cross validation sensitivity
        folds = range(min_k,kfolds+1)
        # List to store all metrics per kfold cross validation run
        cv_results = {f'kfold_{fold}':{} for fold in folds}
        tolerance = 0.001