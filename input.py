##########################################################################
#### Raw CSV data processing and PCA dimensionality reduction
#### Author : Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import pandas as pd
import os
import configparser

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

class DataReader(PathConfig):

    def __init__(self, file_paths):
        super().__init__()
        self._file_paths = file_paths
        
    def combine_csv(self):
        pass

class DataProcessor(PathConfig):

    def __init__(self):
        super().__init__()

class DataPackager(PathConfig):

    def __init__(self):
        super().__init__()

def main():
    pass

if __name__ == "__main__":
    main()