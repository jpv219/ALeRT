##########################################################################
### Further sample selection guided by decision tree
### Author: Fuyue Liang and Juan Pablo Valdes
### Department of Chemical Engineering, Imperial College London
##########################################################################

import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import subprocess
# For the models
from sklearn.tree import _tree
from data_utils import DataLoader
from model_lib import ModelConfig
# For model visualization
import matplotlib.pyplot as plt
# For path config
from paths import PathConfig
from abc import ABC, abstractmethod

PATH = PathConfig()

class ActLearSampler(ABC,PathConfig):
    def __init__(self, case):
        self.case = case
        super().__init__()

    @abstractmethod
    def generate_rules(self):
        pass

class GSX_Sampling(ActLearSampler):
    def __init__(self, case):
        super().__init__(case)
    
    def generate_rules(self, X_df):
        return super().generate_rules()


class DT_Sampling(ActLearSampler):

    def __init__(self,case) -> None:
        super().__init__(case)
        # Model configurer
        self.model_config = ModelConfig()
        
        # selecting corresponding wrapper, hyperparams and model_name
        self.wrapper_model = self.model_config.get_wrapper('dt')
        self.model_params = self.model_config.get_hyperparameters('dt')
        self.model_name = self.model_config.get_model_name('dt')

    def load_dt_model(self):

        # Instantiating the wrapper with the corresponding hyperparams
        model_instance = self.wrapper_model(**self.model_params)

        best_model_path = os.path.join(self.bestmodel_savepath, self.model_name,'best_model.pkl')

        if os.path.exists(best_model_path):
            # Load best trained model
            dt_model = model_instance.load_model(best_model_path, is_mlp = False)

        else:
            print('-'*72)
            print('Decision Tree model has not been trained yet, entering reg_train routine...')
            print('-'*72)
            
            #train model
            self.train_dt_model()
            #load trained model
            dt_model = model_instance.load_model(best_model_path, is_mlp = False)

        return dt_model
    
    def train_dt_model(self):
        
        #reg_train.py inputs to train a fresh dt if best model does not exist
        # asking for: first kfold, with sens?, hyperparam tune? final kfold with s?
        inputs = [self.case, 'dt','y','y','y','y']

        # Concatenate inputs into a single string separated by newline
        input_str = "\n".join(inputs)

        # Provide input using subprocess.Popen
        proc = subprocess.Popen(['python', 'reg_train.py'], stdin=subprocess.PIPE)
        proc.communicate(input=input_str.encode())  # Encode input string to bytes
    
    @staticmethod
    def extract_rules(tree, feature_names, scaler_folder):
        tree_ = tree.tree_ # stores the entire binary tree structure, represented as a number of parallel array
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
            for i in tree_.feature
        ]

        paths = []
        path = []

        def recurse(scaler_folder, node, path, paths):
            '''
            The i-th element of each array holds information about the node i.
            Among these arrays, we have:
            childen_left[i]: id of the left child of node i or -1 if leaf node
            childen_right[i]: id of the right child of node i or -1 if leaf node
            feature[i]: feature used for splitting node i
            threshold[i]: threshold value at node i
            
            '''
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                scaled_threshold = tree_.threshold[node].reshape(-1,1)
                # load the corresponding scaler for the feature: split name for file saving problem
                with open(os.path.join(scaler_folder,f'scaler_{name.split()[0]}.pkl'),'rb') as f:
                    scaler = pickle.load(f)
                # scale the threshold value back to its original ranges and convert back to type of numpy.float64
                threshold = scaler.inverse_transform(scaled_threshold).reshape(-1)[0]
                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(scaler_folder,tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(scaler_folder,tree_.children_right[node], p2, paths)
            else:
                path += [(node, tree_.n_node_samples[node], np.round(tree_.impurity[node],4))]
                paths += [path]
        
        recurse(scaler_folder, 0, path, paths)

        # sort by node impurity (mean squared error)
        mse = [p[-1][-1] for p in paths]
        ii = list(np.argsort(mse))
        paths = [paths[i] for i in reversed(ii)]

        rules = []
        for path in paths:
            rule = "splitting rules: "
            
            for p in path[:-1]:
                if rule != "splitting rules: ":
                    rule += " and "
                rule += str(p)
            rule += f" | MSE: {path[-1][-1]}"
            rules += [rule]

        return rules

    def generate_rules(self, X_df):
        '''
        return the sample space (splitting rules) for resampling
        '''

        # initialize model instance
        model = self.load_dt_model()

        scaler_folder = os.path.join(self.input_savepath, self.case, 'ini')

        # extract the splitting rules in the order of high to low MSE
        rules = self.extract_rules(model, X_df.columns, scaler_folder)

        # Save the feature importantce
        fi_df = pd.DataFrame(columns=X_df.columns)
        fi_df.loc[len(fi_df)] = model.feature_importances_
        fi_df_T = fi_df.transpose()
        fi_df_T = fi_df_T.sort_values(by=fi_df_T.columns[0],ascending=False)
        fi_df_T.to_pickle(os.path.join(self.resample_savepath,self.case,'dt','log_rules',f'resample_FI.pkl'))

        # Visualize the feature important and save the plot
        fig = plt.figure(figsize=(8,6))
        plt.bar(fi_df_T.index, fi_df_T[0])
        plt.xlabel(r'Geometry Parameters', fontsize=20)
        plt.ylabel(r'Feature Importance', fontsize=20)
        plt.xticks(rotation=45)
        fig.savefig(os.path.join(self.fig_savepath,self.case,'dt',f'{self.case}_FI.png'),dpi=200)
        plt.show()

        return rules

def main():

    case = input('Select a study from where to load proccessed data packs (sp(sv)_geom): ')

    sampler_choice = input('Select AL sampling technique to generate guided sample space to explore (dt, gsx): ')

    AL_samplers = {'dt': DT_Sampling(case),
                   'gsx': GSX_Sampling(case)}
    
    sampler = AL_samplers.get(sampler_choice)

    dataloader = DataLoader(case)
    
    inidata_dir = os.path.join(PATH.input_savepath, case, 'ini')

    data_packs = dataloader.load_packs(inidata_dir)

    # Input the extracted data
    X_ini_df = data_packs[0]

    rules = sampler.generate_rules(X_ini_df)
    
    # store rules to local log file
    with open(os.path.join(PATH.resample_savepath,case,'dt','log_rules',f'{sampler_choice}_rules.log'), 'w') as file:
        for r in rules:
            file.write(r+'\n')
            print(r)
    

if __name__ == "__main__":
    main()