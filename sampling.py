### Further sample selection guided by decision tree
### Author: Fuyue Liang
### Department of Chemical Engineering, Imperial College London

import pandas as pd
import numpy as np
import os
import pickle
# For the models
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import _tree
from data_utils import DataLoader
# For model visualization
import matplotlib.pyplot as plt
# For path config
from paths import PathConfig

PATH = PathConfig()

def DT_extract_rules(tree, feature_names, scaler_folder):
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

def DT_guided_resam(case, X_df, y_df,resample_savepath, fig_folder, scaler_folder):
    '''
    return the sample space (splitting rules) for resampling
    '''

    X_data = X_df.to_numpy()
    y_data = y_df.to_numpy()

    # initialize model instance
    model = DecisionTreeRegressor(max_depth=5, random_state=2024)

    # Fit the model
    model.fit(X_data,y_data)

    # extract the splitting rules in the order of high to low MSE
    rules = DT_extract_rules(model, X_df.columns, scaler_folder)

    # Save the feature importantce
    fi_df = pd.DataFrame(columns=X_df.columns)
    fi_df.loc[len(fi_df)] = model.feature_importances_
    fi_df_T = fi_df.transpose()
    fi_df_T = fi_df_T.sort_values(by=fi_df_T.columns[0],ascending=False)
    fi_df_T.to_pickle(os.path.join(resample_savepath,case,f'resample_FI.pkl'))

    # Visualize the feature important and save the plot
    fig = plt.figure(figsize=(8,6))
    plt.bar(fi_df_T.index, fi_df_T[0])
    plt.xlabel(r'Geometry Parameters', fontsize=20)
    plt.ylabel(r'Feature Importance', fontsize=20)
    plt.xticks(rotation=45)
    fig.savefig(os.path.join(fig_folder,f'{case}_FI.png'),dpi=200)
    plt.show()

    # Visualize and save the tree
    fig = plt.figure(figsize=(8,6))
    tree.plot_tree(model,
                feature_names=X_df.columns.values,
                proportion=True,
                rounded=True,
                filled=True)
    fig.savefig(os.path.join(fig_folder,f'{case}_tree.png'),dpi=600)

    return model, rules

def main():

    case = input('Select a study to process raw datasets (sp_(sv)geom, (sv)surf, (sv)geom): ')

    dataloader = DataLoader(case)

    inidata_dir = os.path.join(PATH.input_savepath, case, 'ini')
    scaler_dir = os.path.join(PATH.input_savepath, case)

    data_packs = dataloader.load_packs(inidata_dir)

    # Input the extracted data
    X_ini_df, y_ini_df = data_packs[0], data_packs[1]

    regr, rules = DT_guided_resam(case,X_ini_df,y_ini_df,PATH.resample_savepath,PATH.fig_savepath,scaler_dir)
    
    # store rules to local log file
    with open(os.path.join(PATH.resample_savepath,case,f'resample_rules.log'), 'w') as file:
        for r in rules:
            file.write(r+'\n')
            print(r)
    

if __name__ == "__main__":
    main()