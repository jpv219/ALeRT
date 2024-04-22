##########################################################################
#### Raw CSV data processing and PCA dimensionality reduction
#### Author : Juan Pablo Valdes and Fuyue Liang
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from paths import PathConfig
from data_utils import DataReader, DataProcessor, DataPackager


def preprocess_data(df: pd.DataFrame, dt_processor:DataProcessor, in_idx, out_idx, 
                    filter_range, scale_choice):
    
    """
    Preprocesses the input DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        in_idx (int): Cut-off index between input and output params.
        out_idx (str): Selected output parameters idx to include.
        filter_range (str): Filter percentages [0,1] for min/max cases.
        scale_choice (str): Scaling method (norm/log/robust/power/quantile).

    Returns:
        tuple: Tuple containing scaled input and output features.
    """

    param_idx = [(idx,param) for idx,param in enumerate(df.columns)]
    
    #drop output columns assuming DOE df is concatenated first always
    X_df = df.drop(df.columns[int(in_idx):], axis = 1)

    # Drop output columns
    if out_idx == 'all':
        y_df = df.drop(df.columns[:int(in_idx)], axis=1)

    else:
        #selected variables to preserve
        out_idx_list = [int(x) for x in out_idx.split(',')]

        # Raise exception if chosen idx are located within the selected input features
        for idx in out_idx_list:
            if idx < int(in_idx):
                raise ValueError(f'idx = {idx} selected is not within the output features idxs: {in_idx} to {param_idx[-1][0]} ')
            
        y_df = df[df.columns[out_idx_list]].copy()

    # Filter cases with min/max feature values
    percentages = [float(x) for x in filter_range.split(',')]
    if len(percentages) < 2:
        raise ValueError('Either min or max filter percentage was not defined')
    
    X_minmax, y_minmax, X_filtered, y_filtered = dt_processor.filter_minmax([X_df, y_df], bottom=percentages[0], upper=percentages[1])

    # Scale input and output features
    X_scaled = dt_processor.scale_data([X_df.copy(), X_minmax, X_filtered], scaling=scale_choice)
    y_scaled = dt_processor.scale_data([y_df.copy(), y_minmax, y_filtered], scaling=scale_choice)

    # plot datapack with filtered minmax cases
    dt_processor.plot_scaling(X_df,X_scaled[-1],X_scaled[1],data_label='inputs')
    dt_processor.plot_scaling(y_df,y_scaled[-1],y_scaled[1],data_label='outputs')

    return X_scaled, y_scaled

def process_ini(df, X_scaled, y_scaled, dt_processor: DataProcessor, dt_packager: DataPackager):
    """
    Process initial data.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        X_scaled (pd.DataFrame): Scaled input features.
        y_scaled (pd.DataFrame): Scaled output features.
        dt_processor (DataProcessor): Data processor instance.
        dt_packager (DataPackager): Data packager instance.
    """

    # Case splitting for sampling comparison, setting an initial set to train with and explore: AL vs. Random
    X_ini, X_random, y_ini, y_random = train_test_split(X_scaled[-1], y_scaled[-1], test_size=0.2, random_state=2024)
    
    # train test splitting with filtered datapack from the initial dataset to be used
    X_train, X_test, y_train, y_test = train_test_split(X_ini, y_ini, test_size=0.25, random_state=2024)
    print(f'Sizes of ini training: {X_train.shape[0]}; test: {X_test.shape[0]}; random: {X_random.shape[0]} ')

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

        # Package data for further use training and deploying regression models
        data_pack = [df,X_train,y_train_reduced,X_test,y_test_exp,y_train,pca_info_df]
        labels = ['full','X_train_i','y_train_i_red','X_test_i','y_test_i','y_train_i_raw','PCA_info']

    else:
        # Expand y_train and test columns containing arrays to individual columns per feature value for correct handling by regressor
        y_train_exp = dt_packager.expand_targets(y_train)

        # Package data for further use training and deploying regression models
        data_pack = [df,X_train,y_train_exp,X_test,y_test_exp,y_train]
        labels = ['full','X_train_i','y_train_i','X_test_i','y_test_i','y_train_i_raw']
    
    # Package initial data sets
    dt_packager.package_data(data_pack,labels, datasample= 'ini')

    # Package random data sets
    random_pack = [X_random, y_random]
    random_labels = ['X_random', 'y_random_raw']

    dt_packager.package_data(random_pack, random_labels, datasample='random')

def process_dt(df:pd.DataFrame, X_scaled, y_scaled, dt_packager: DataPackager):
    """
    Process resampled data following decision tree (dt) active sampling.

    Parameters:
        df (pd.DataFrame): Input DataFrame from dt csvs.
        X_scaled (pd.DataFrame): Scaled input features.
        y_scaled (pd.DataFrame): Scaled output features.
        dt_packager (DataPackager): Data packager instance.
    """

    # train test splitting with filtered datapack from the initial dataset to be used
    X_train, y_train = X_scaled[-1], y_scaled[-1]

    print(f'Sizes of dt training set: {X_train.shape[0]} ')

    # Package data for further use training and deploying regression models
    data_pack = [df,X_train,y_train]
    labels = ['full','X_train_dt','y_train_dt_raw']
    
    # Package initial data sets
    dt_packager.package_data(data_pack,labels, datasample= 'dt')

def main():
    
    case_name = input('Select a study to process raw datasets (sp_(sv)geom): ')

    data_name = input('Select a dataset to process from the study selected above (ini, dt): ')
       
    dt_reader = DataReader(case_name,data_name)
    dt_processor = DataProcessor(case_name, data_name)
    dt_packager = DataPackager(case_name, data_name)
    # Global paths configuration
    PATH = PathConfig()
    
    #Combine csv and DOE label files
    df = dt_reader.combine_data()
    
    # Dropping SMX_pos, length (sm) and Height, arc_length, Time (sv) columns from input features required by the regressor
    columns_to_drop = ['SMX_pos (mm)', 'Length', 'Height', 'arc_length', 'Time']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns_to_drop, axis=1)
    
    # Divide between input and output params in the combined df
    params = df.columns
    param_idx = [(idx,param) for idx,param in enumerate(params)]

    print(f'The input and output parameters in this case study are: {param_idx}')

    if data_name == 'ini':
        in_idx = input('Provide cut-off index between input and output params (first out idx): ')

        #Choose idx for output variables
        out_idx = input('Select the output parameters idx to include (separated by ,) or choose \'all\': ')

        # Filter cases with min/max feature values
        filter_range = input('Define filter percentages [0,1] for min/max cases (default: min_filter,max_filter = 0,0): ')

        #Scale input and output features
        scale_choice = input('Select a scaling method (norm/log/robust/power/quantile): ')

        X_scaled, y_scaled = preprocess_data(df, dt_processor, in_idx, out_idx, filter_range, scale_choice)
    
        process_ini(df, X_scaled, y_scaled, dt_processor, dt_packager)

    elif data_name == 'dt':
        # extract the columns from ini pre-processing
        scaler_path = os.path.join(PATH.input_savepath,case_name, 'ini')
        file_names = os.listdir(scaler_path)
        # filter the columns for X_df
        X_columns = [col.split('scaler_X_')[1].split('.')[0] for col in file_names if col.startswith('scaler_X_')]
        regex_pattern = '|'.join(X_columns)
        X_df = df.filter(regex=regex_pattern)
        # filter the columns for X_df
        y_columns = [col.split('scaler_y_')[1].split('.')[0] for col in file_names if col.startswith('scaler_y_')]
        y_df = df[y_columns]

        X_scaled = dt_processor.scale_data([X_df.copy()], scaling=None)
        y_scaled = dt_processor.scale_data([y_df.copy()], scaling=None)

        process_dt(df, X_scaled, y_scaled, dt_packager)


if __name__ == "__main__":
    main()