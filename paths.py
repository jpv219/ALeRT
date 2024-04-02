##########################################################################
#### Model train and evaluate utilities
#### Author : Juan Pablo Valdes and Fuyue Liang
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import configparser
import os

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
    def pca_savepath(self):
        return self._config['Path']['pca']

    @property
    def raw_datapath(self):
        return self._config['Path']['csv']
    
    @property
    def label_datapath(self):
        return self._config['Path']['doe']
    
    @property
    def model_savepath(self):
        return self._config['Path']['models']
