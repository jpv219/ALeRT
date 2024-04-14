##########################################################################
#### Raw CSV data processing and PCA dimensionality reduction
#### Author : Juan Pablo Valdes and Fuyue Liang
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from data_utils import DataReader, DataProcessor, DataPackager

def main():
    
    case_name = input('Select a study to process raw datasets (sp_(sv)geom, (sv)surf, (sv)geom): ')

    # Class instances
    dt_reader = DataReader(case_name)
    dt_processor = DataProcessor(case_name)
    dt_packager = DataPackager(case_name)

    #Combine csv and DOE label files
    df = dt_reader.combine_data()

    # Dropping SMX_pos, length (sm) and Height, arc_length, Time (sv) columns from input features required by the regressor
    for column in df.columns:
        if column == 'SMX_pos (mm)' or column == 'Length' or column == 'Height' or column == 'arc_length' or column == 'Time':
            df = df.drop(column,axis=1)

    # Divide between input and output params in the combined df
    params = df.columns
    param_idx = [(idx,param) for idx,param in enumerate(params)]

    print(f'The input and output parameters in this case study are: {param_idx}')

    in_idx = input('Provide cut-off index between input and output params (first out idx): ')

    #drop output columns assuming DOE df is concatenated first always
    X_df = df.drop(df.columns[int(in_idx):], axis = 1)

    #Choose idx for output variables
    out_idx = input('Select the output parameters idx to include (separated by ,) or choose \'all\': ')

    if out_idx == 'all':
        y_df = df.drop(df.columns[:int(in_idx)], axis = 1)

    else:
        #selected variables to preserve
        out_idx_list = [int(x) for x in out_idx.split(',')]
        
        # Raise exception if chosen idx are located within the selected input features
        for idx in out_idx_list:
            if idx < int(in_idx):
                raise ValueError(f'idx = {idx} selected is not within the output features idxs : {in_idx} to {param_idx[-1][0]} ')

        y_df = df[df.columns[out_idx_list]].copy()

    # # Filter cases with min/max feature values
    percentage_choice = input('Define filter percentages [0,1] for min/max cases (default: min_filter,max_filter = 0,0): ')
    percentages = [float(x) for x in percentage_choice.split(',')]

    if len(percentages) < 2:
        raise ValueError('Either min or max filter percentage was not defined')

    X_minmax, y_minmax, X_filtered, y_filtered = dt_processor.filter_minmax([X_df,y_df],bottom=percentages[0],upper=percentages[1])
    
    #Scale input and output features
    scale_choice = input('Select a scaling method (norm/log/robust/power/quantile): ')

    # scale data pack containing [original data, minmax found values, data w/o filtered cases]
    X_scaled = dt_processor.scale_data([X_df.copy(),X_minmax,X_filtered],scaling=scale_choice)
    y_scaled = dt_processor.scale_data([y_df.copy(),y_minmax,y_filtered],scaling=scale_choice)

    # plot datapack with filtered minmax cases
    dt_processor.plot_scaling(X_df,X_scaled[-1],X_scaled[1],data_label='inputs')
    dt_processor.plot_scaling(y_df,y_scaled[-1],y_scaled[1],data_label='outputs')

    # Case splitting for sampling comparison, setting an initial set to train with and explore: AL vs. Random
    X_ini, X_random, y_ini, y_random = train_test_split(X_scaled[-1], y_scaled[-1], test_size=0.2, random_state=2024)
    
    # train test splitting with filtered datapack from the initial dataset to be used
    X_train, X_test, y_train, y_test = train_test_split(X_ini, y_ini, test_size=0.25, random_state=2024)

    # recombine filtered minmax cases into initial training data pack
    combine_choice = input('Include the filtered cases into training? (y/n):')
    if combine_choice.lower() == 'y':
        X_train = pd.concat([X_train,X_scaled[1]],axis=0)
        y_train = pd.concat([y_train,y_scaled[1]],axis=0)

    # Expand y_test arrays into separate columns for further regression eval
    y_test_exp = dt_packager.expand_targets(y_test)

    pca_choice = input('Carry out dimensionality reduction through PCA? (y/n): ')

    if pca_choice.lower() == 'y':

        # Carry out PCA on scaled outputs for training only
        var_ratio = 0.95
        y_train_reduced, pca_info_df = dt_processor.PCA_reduction(y_train,var_ratio, datasample='ini')

        y_random_reduced, pca_info_reduced = dt_processor.PCA_reduction(y_random, var_ratio,datasample='random')

        # Package data for further use training and deploying regression models
        data_pack = [df,X_train,y_train_reduced,X_test,y_test_exp,pca_info_df]
        labels = ['full','X_train_i','y_train_i_red','X_test_i','y_test_i','PCA_info']

        random_pack = [X_random, y_random_reduced,pca_info_reduced]
        random_labels = ['X_random', 'y_random','PCA_info']

    else:

        # Expand y_train and test columns containing arrays to individual columns per feature value for correct handling by regressor
        y_train_exp = dt_packager.expand_targets(y_train)

        # Expand targets for random dataset split for further model train and eval
        y_random_exp = dt_packager.expand_targets(y_random)

        # Package data for further use training and deploying regression models
        data_pack = [df,X_train,y_train_exp,X_test,y_test_exp]
        labels = ['full','X_train_i','y_train_i','X_test_i','y_test_i']

        random_pack = [X_random, y_random_exp]
        random_labels = ['X_random', 'y_random']
    
    # Package initial data sets
    dt_packager.package_data(data_pack,labels, datasample= 'ini')

    # Package random data sets
    dt_packager.package_data(random_pack, random_labels, datasample='random')


if __name__ == "__main__":
    main()