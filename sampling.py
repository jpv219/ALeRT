### Further sample selection guided by decision tree
### Author: Fuyue Liang
### Department of Chemical Engineering, Imperial College London

import pandas as pd
import numpy as np
import os
# For the models
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import _tree
# from numpy import absolute
# from numpy import mean
# from numpy import std
# For model visualization
import matplotlib.pyplot as plt
# For path config
from paths import PathConfig


def DT_extract_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
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

def DT_guided_resam(case, X_df, y_df,resample_savepath, fig_folder):
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
    rules = DT_extract_rules(model, X_df.columns)

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
    path = PathConfig()
    case = input('Select a study to process raw datasets (sp_(sv)geom, (sv)surf, (sv)geom): ')
    label_package = []
    data_packs = []

    # Read package names to later import
    with open(os.path.join(path.input_savepath,case,'Load_Labels.txt'), 'r') as file:
        lines = file.readlines()

        for line in lines:
            label_package.append(line.split('\n')[0])
 
    # Save only train and test packs
    label_package =  [item for item in label_package if item not in ['full', 'PCA_info']]
    
    # Load pickle files
    for label in label_package:

        data_path = os.path.join(path.input_savepath,case,f'{label}.pkl')

        if os.path.exists(data_path):

            data_pack = pd.read_pickle(data_path)          
            data_packs.append(data_pack)

    # Input the extracted data
    X_ini_df, y_ini_df = data_packs[0], data_packs[1]

    regr, rules = DT_guided_resam(case,X_ini_df,y_ini_df,path.resample_savepath,path.fig_savepath)
    
    # store rules to local log file
    with open(os.path.join(path.resample_savepath,case,f'resample_rules.log'), 'w') as file:
        for r in rules:
            file.write(r+'\n')
            print(r)
    

if __name__ == "__main__":
    main()