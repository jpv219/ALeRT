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
        super().__init__()

        self.case = case
        self.scaler_folder = os.path.join(self.input_savepath, self.case, 'ini')

    @abstractmethod
    def generate_rules(self):
        pass

############################################################################################################################

class GSX_Sampling(ActLearSampler):
    def __init__(self, case,num_samples):
        super().__init__(case)
        self.num_samples = num_samples

    def inverse_transform_scaling(self, X_df: pd.DataFrame) -> pd.DataFrame:

        inv_X_df = pd.DataFrame(columns = X_df.columns, index = X_df.index)
        
        for column in X_df.columns:
        
            # load the corresponding scaler for the feature: split name for file saving problem
            with open(os.path.join(self.scaler_folder,f'scaler_X_{column.split()[0]}.pkl'),'rb') as f:
                scaler = pickle.load(f)
            
            # scale the threshold value back to its original ranges and convert back to type of numpy.float64
            threshold = scaler.inverse_transform(X_df[column].values.reshape(-1,1))

            inv_X_df[column] = threshold
        
        return inv_X_df

    def generate_rules(self, datapacks, savepath):
        
        X_df = datapacks[0]
        
        inv_X_df = self.inverse_transform_scaling(X_df)
        
        selected_indices = []

        # Select one random case as initial sample
        initial_index = 35#np.random.randint(0, len(inv_X_df))
        selected_indices.append(initial_index)

        while len(selected_indices) < self.num_samples:
            # Get indices of cases that are not yet selected
            not_selected_indices = [i for i in range(len(inv_X_df)) if i not in selected_indices]

            min_distances = []
            for i in not_selected_indices:
                # Calculate distances between current case and selected cases
                distances = [np.linalg.norm(inv_X_df.iloc[i] - inv_X_df.iloc[j]) for j in selected_indices]
                min_distances.append(min(distances))

            # Select the case with maximum minimum distance
            new_index = not_selected_indices[np.argmax(min_distances)]
            selected_indices.append(new_index)

        # Collect selected samples outside the loop
        selected_samples = inv_X_df.iloc[selected_indices]

        selected_samples.to_pickle(os.path.join(savepath,'gsx_df.pkl'))

        rules = ['Cases to sample from GSx',str(list(selected_samples.columns))]

        # Append samples selected as separate string lines to the rules list
        for i in range(len(selected_samples)):

            rounded_values = [round(val, 4) if isinstance(val, float) else val for val in selected_samples.iloc[i].values]

            rules.append(str(rounded_values))

        # Append min max values from the new sample space generated
        rules.append('Max/min values per input feature')

        for feature in selected_samples.columns:

            rules.append(str(feature) + '[MAX, MIN]')
            rules.append(str([selected_samples[feature].max(), selected_samples[feature].min()]))

        return rules

class DT_Sampling(ActLearSampler):

    def __init__(self,case) -> None:
        super().__init__(case)
        # Model configurer
        self.model_config = ModelConfig()
        
        # selecting corresponding wrapper, hyperparams and model_name
        self.wrapper_model = self.model_config.get_wrapper('dt')
        self.model_params = self.model_config.get_hyperparameters('dt')
        self.model_name = self.model_config.get_model_name('dt')

    def load_dt_model(self,load_model:str,data_packs):

        # Instantiating the wrapper with the corresponding hyperparams
        model_instance = self.wrapper_model(**self.model_params)

        dt_model = None
        best_model_path = os.path.join(self.bestmodel_savepath, self.model_name, 'best_model.pkl')

        if load_model == 'y' and os.path.exists(best_model_path):
            # Load best trained model
            dt_model = model_instance.load_model(best_model_path, is_mlp=False)
        else:
            print('-' * 72)
            
            if load_model == 'y':
                print('Decision Tree model has not been trained yet, entering reg_train routine...')
            else:
                print('Entering simple reg_train routine for DT model...')
            
            print('-' * 72)

            # Train model
            dt_model = self.train_dt_model(data_packs)

        return dt_model
    
    def train_dt_model(self,data_packs):
        
        # train dt model with different augemented data
        # data packs
        dataloader = DataLoader(self.case)
        data_choice = 'dt'
        pca = dataloader.pca

        # Instantiating the wrapper with the corresponding hyperparams
        model_instance = self.wrapper_model(**self.model_params)

        # Getting regressor object from wrapper
        model = model_instance.init_model()

        # Train model with new data, w/o further tuning or cross validation
        cv_options = {'do_kfold': False,
            'ksens' : False,
            'do_hp_tune': False}
        
        trained_dt_model = model_instance.model_train(data_packs, model,
                                cv_options, self.model_name)
        
        # Calling model evaluate with tuned model
        model_instance.model_evaluate(trained_dt_model, self.model_name, data_packs,
                                    self.case,pca, data_choice)
        
        return trained_dt_model
    
    def extract_rules(self, tree, feature_names):
        tree_ = tree.tree_ # stores the entire binary tree structure, represented as a number of parallel array
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
            for i in tree_.feature
        ]

        paths = []
        path = []

        def recurse(node, path, paths):
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
                with open(os.path.join(self.scaler_folder,f'scaler_X_{name.split()[0]}.pkl'),'rb') as f:
                    scaler = pickle.load(f)

                # scale the threshold value back to its original ranges and convert back to type of numpy.float64
                threshold = scaler.inverse_transform(scaled_threshold).reshape(-1)[0]

                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)

            else:
                path += [(node, tree_.n_node_samples[node], np.round(tree_.impurity[node],4))]
                paths += [path]
        
        recurse(0, path, paths)

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

    def generate_rules(self, data_packs, savepath):
        '''
        return the sample space (splitting rules) for resampling
        '''

        load_model = input('Load pre-trained model (if available)? (y/n): ')
        
        # initialize model instance
        model = self.load_dt_model(load_model,data_packs)

        # extract the splitting rules in the order of high to low MSE
        rules = self.extract_rules(model, data_packs[0].columns)

        # Save the feature importantce
        fi_df = pd.DataFrame(columns=data_packs[0].columns)
        fi_df.loc[len(fi_df)] = model.feature_importances_
        fi_df_T = fi_df.transpose()
        fi_df_T = fi_df_T.sort_values(by=fi_df_T.columns[0],ascending=False)
        fi_df_T.to_pickle(os.path.join(savepath,f'resample_FI.pkl'))

        # Visualize the feature important and save the plot
        fig = plt.figure(figsize=(8,6))
        plt.bar(fi_df_T.index, fi_df_T[0])
        plt.xlabel(r'Geometry Parameters', fontsize=20)
        plt.ylabel(r'Feature Importance', fontsize=20)
        plt.xticks(rotation=45)
        fig.savefig(os.path.join(self.fig_savepath,self.case,'dt',f'Feature_Importance.png'),dpi=200)
        plt.show()

        return rules

def main():

    case = input('Select a study from where to load proccessed data packs (sp_(sv)geom): ')

    sampler_choice = input('Select AL sampling technique to generate guided sample space to explore (dt, gsx): ')

    AL_samplers = {'dt': DT_Sampling(case),
                   'gsx': GSX_Sampling(case, num_samples=100)}
    
    sampler = AL_samplers.get(sampler_choice)

    dataloader = DataLoader(case)
    
    inidata_dir = os.path.join(PATH.input_savepath, case, 'ini')
    random_dir = os.path.join(PATH.resample_savepath,case,'random')
    resample_dir = os.path.join(PATH.resample_savepath, case, sampler_choice)

    inidata_packs = dataloader.load_packs(inidata_dir)
    random_packs = dataloader.load_packs(random_dir)

    # get all the available trials in the resample folder 
    trial_list = [f.name for f in os.scandir(resample_dir) if f.is_dir()]
    
    # generate the new trial folder
    new_trial_folder = os.path.join(resample_dir,f'trial{len(trial_list)+1}')
    if not os.path.exists(new_trial_folder):
        os.makedirs(new_trial_folder)
    
    if sampler_choice == 'gsx':
        print('-'*72)
        print(f'Running GSx sampling')
        print('-'*72)
        rules = sampler.generate_rules(random_packs,new_trial_folder)

        # extract the gsx cases based on the rules (indexes) from the random cases
        dfgsx = pd.read_pickle(os.path.join(new_trial_folder,'gsx_df.pkl'))
        dfx = pd.read_pickle(os.path.join(random_dir,'X_random.pkl'))
        dfy = pd.read_pickle(os.path.join(random_dir,'y_random_raw.pkl'))

        dfxgsx = dfx.loc[dfgsx.index]
        dfygsx = dfy.loc[dfgsx.index]

        dfxgsx.to_pickle(os.path.join(resample_dir,'X_train_gsx.pkl'))
        dfygsx.to_pickle(os.path.join(resample_dir,'y_train_gsx_raw.pkl'))
    
    elif len(trial_list) == 0 and sampler_choice == 'dt':
        print('-'*72)
        print(f'No previous sampling detected. Using initial training dataset for sampling now ...')
        print('-'*72)
        rules = sampler.generate_rules(inidata_packs,new_trial_folder)
    else:
        print('-'*72)
        print(f'Sampling rules generating for {sampler_choice} Trial {len(trial_list)+1} ...')
        print('-'*72)

        aug_packs = dataloader.augment_data(sampler_choice)
        rules = sampler.generate_rules(aug_packs,new_trial_folder)

    # store rules to local log file
    with open(os.path.join(new_trial_folder,f'{sampler_choice}_rules.log'), 'w') as file:
        for r in rules:
            file.write(r+'\n')
            print(r)
    

if __name__ == "__main__":
    main()
