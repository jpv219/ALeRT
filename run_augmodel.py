##########################################################################
#### Augmented trained model run and evaluation
#### Author : Juan Pablo Valdes and Fuyue Liang
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import os
from paths import PathConfig
import pandas as pd

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
    label_package =  [item for item in label_package if item not in ['full', 'PCA_info']]
    
    # Load pickle files
    for label in label_package:

        data_path = os.path.join(path.input_savepath,case,f'{label}.pkl')

        if os.path.exists(data_path):

            data_pack = pd.read_pickle(data_path)          
            data_packs.append(data_pack)


    model_choice = input('Select trained model to load and deploy: (dt, xgb, rf, svm, knn, mlp_br, mlp): ')