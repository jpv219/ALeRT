##########################################################################
#### Raw CSV data processing and PCA dimensionality reduction
#### Author : Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import pandas as pd
import numpy as np
import ast
import os
import pickle
import configparser
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

    def __init__(self,case):
        super().__init__()
        self._case = case

    def collect_csv_pkl(self):
        
        csv_list = []
        DOE_list = []

        # data and DOE paths for the case under consideration
        csv_path = os.path.join(self.raw_datapath,self._case) #output data
        doe_path = os.path.join(self.label_datapath,self._case) #input data/labels

        file_count = len(os.listdir(csv_path))

        # Loop concatenating DOE and csv files from different parametric runs
        for i in range(1,file_count+1):

            csv_filename = f'{self._case}_{i}.csv'
            doe_filename = f'LHS_{self._case}_{i}.pkl'
            
            # Read csv and DOE files and store in corresponding list
            data = pd.read_csv(os.path.join(csv_path, csv_filename))
            csv_list.append(data)
            
            doe_lbl = pd.read_pickle(os.path.join(doe_path, doe_filename))
            DOE_list.append(doe_lbl)

        # Concatenate all files into a single df
        data_df = pd.concat(csv_list, ignore_index=True)
        data_df = data_df.set_index('Run_ID', drop=False)

        DOE_df = pd.concat(DOE_list, ignore_index=True)

        # Count how many runs are stored successfully in the CSV.
        run_list = set(data_df['Run_ID'].tolist())

        return data_df, DOE_df, run_list

    def combine_data(self):
        
        data_df, DOE_df, run_list = self.collect_csv_pkl()
        
        #Sort runs by run number
        sorted_runs = sorted(run_list, key=lambda x: int(x.split('_')[-1]))

        # Filter out from DOE labels only the cases that were finished successfully.
        # Run_Id number at the end of the string is matched to the DOE_df index, considering run 1 is index 0
        df_DOE_filtered = DOE_df[DOE_df.index.isin([int(run.split('_')[-1])-1 for run in sorted_runs])]

        # Create an index column and set it as new index, with ID number equal to Run_ID-1
        data_df['index'] = data_df.index.str.split('_').str[-1].astype(int) -1

        # Ordered csv df by Run_IDnumber and drop Run_ID column
        data_df_sorted = data_df.sort_values(by='index').set_index('index')
        data_df_sorted = data_df_sorted.drop(columns=['Run_ID'])

        #Re-shaping dataframe into object arrays per index, grouping run values into lists
        if 'sv' in self._case:
            data_df_reshaped = data_df_sorted
        else:
            data_df_reshaped = data_df_sorted.groupby('index').agg(lambda x: x.tolist())

        # Merge input parameters (labels) with hydrodynamic data output (target)
        df = pd.concat([df_DOE_filtered,data_df_reshaped],axis=1)

        for column in df.columns:

                # Apply scaler, reshaping into a column vector (n,1) for scaler to work if output feature is an array
                if df[column].dtype == 'object':
                    #Convert text lists into np arrays
                    df[column] = df[column].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))
                else:
                    df[column] = df[column].astype(float)

        return df
    
class LogScaler:
    def __init__(self,base=10):
        self.base = base
        self.log_base = np.log(self.base)
    def fit(self, X, y=None):
        # No operation needed during fitting
        return self
    def transform(self,X):
        return np.log(X) / self.log_base
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X) 

class DataProcessor(PathConfig):

    def __init__(self,case):
        super().__init__()
        self._case = case

    def find_indices(self,df_column,where,percentage=0.0):
        '''
        Find the index with min and max values across all arrays in a column
        '''
        indices_in_df = []
        if percentage == 0.0:
            num_cases = 1
        else:
            # Calculate the number of cases corresponding to the given percentage
            num_cases = max(1,int(percentage*len(df_column)))
            
        if df_column.dtype == 'object':
            for _ in range(num_cases):
                # Concatenate arrays vertically
                stacked_arr = np.vstack(df_column)
                
                # Find the index of the min and max values across all arrays
                if where == 'min':
                    the_index = np.unravel_index(np.argmin(stacked_arr),stacked_arr.shape)
                elif where == 'max':
                    the_index = np.unravel_index(np.argmax(stacked_arr),stacked_arr.shape)
                
                index_in_df = df_column.index[the_index[0]]
                df_column = df_column.drop(index_in_df)
                indices_in_df.append(index_in_df)
        else:
            if where == 'min':
                the_indices = df_column.nsmallest(num_cases).index.tolist()
                indices_in_df = df_column.index[the_indices]
            elif where == 'max':
                the_indices = df_column.nlargest(num_cases).index.tolist()
                indices_in_df = df_column.index[the_indices]
        
        return indices_in_df

    
    def filter_minmax(self, data_pack,bottom=0.0,upper=0.0):
        '''
        Separate the cases with min and max feature values;
        input: both X_df and y_df
        return: X_minmax, X_filtered, y_minmax, y_filtered
        '''
        X_df = data_pack[0]
        y_df = data_pack[1]
        for column in y_df.columns:

            df_min_indices = self.find_indices(y_df[column],'min',bottom)
            df_max_indices = self.find_indices(y_df[column],'max',upper)
            
            # Extract the rows as a Dataframe using doule brackets
            X_minmax = pd.concat([X_df.loc[df_min_indices],X_df.loc[df_max_indices]],axis=0)
            y_minmax = pd.concat([y_df.loc[df_min_indices],y_df.loc[df_max_indices]],axis=0)
            X_filtered = X_df.drop(df_min_indices+df_max_indices)
            y_filtered = y_df.drop(df_min_indices+df_max_indices)

            print(f'Filtering cases of {column}: {len(y_minmax)} out of {len(y_df)}.')

        return X_minmax, y_minmax, X_filtered, y_filtered

    def scale_data(self,data_pack,scaling):

        scaled_data = []
        # scaler = MinMaxScaler(feature_range=(-1,1))
        # Create a Scaler for scaling later
        if scaling == 'norm':
            scaler = MinMaxScaler(feature_range=(-1,1))
        elif scaling == 'log':
            scaler = LogScaler(base=10)
        elif scaling == 'robust':
            scaler = RobustScaler(with_centering=False, with_scaling=True, quantile_range=(25.0,75.0))

        for df in data_pack:

            # Scale output features
            for column in df.columns:

                # Apply scaler, reshaping into a column vector (n,1) for scaler to work if output feature is an array
                if df[column].dtype == 'object':
                    flat_list = [ele_val for ele in df[column] for ele_val in ele]
                    flat_arr = np.array(flat_list).reshape(-1,1)
                    scaler.fit(flat_arr)
                    df[column] = df[column].apply(lambda x: scaler.transform(x.reshape(-1,1)))            
                    # reshaping back to a 1D list
                    df[column] = df[column].apply(lambda x: x.reshape(-1,))
                else:
                    df[column] = scaler.fit_transform(df[column].values.reshape(-1,1))
                    df[column] = df[column].values.reshape(-1,)
            
            if len(data_pack) > 1:
                scaled_data.append(df.copy())
            else:
                scaled_data = df.copy()
            
        return scaled_data
    
    # Visualize data before and after scaling
    def plot_scaling(self, original_data, scaled_data,data_label):
        
        num_features = len(original_data.columns)
        
        fig,ax = plt.subplots(num_features,2, figsize=(12,int(num_features)*5))
        plt.subplots_adjust(hspace=0.8)

        # If only one output feature is selected, reshape the axes to allow plotting
        if num_features == 1:
            ax = ax.reshape((1, 2))

        for i, column in enumerate(original_data.columns):

            # Instructions for output features read as arrays from csv
            if original_data[column].dtype == 'object':

                for j in original_data.index:
                    ax[i,0].plot(original_data[column][j])
                    ax[i,0].set_title(f'Data before: {column}')
                for k in scaled_data.index:
                    ax[i,1].plot(scaled_data[column][k])
                    ax[i,1].set_title(f'Data after: {column}')
            # Scalar input features
            else:
                ax[i,0].plot(original_data[column])
                ax[i,0].set_title(f'Data before: {column}')
                ax[i,1].plot(scaled_data[column])
                ax[i,1].set_title(f'Data after: {column}')

        fig.savefig(os.path.join(self.fig_savepath,f'{self._case}_{data_label}'),dpi=200)
        plt.show()
    
    def PCA_reduction(self,df,var_ratio):
    
        # Empty Dataframe for PCs results and principal axes
        pca_labels = ['PCs', 'Dominant_Features', 'Explained_Var','Principal_Axes']
        pca_df = pd.DataFrame(columns=pca_labels)

        # Carry out expansion and PCA per array features(exclude scalar outputs if any)
        for column in df.select_dtypes(include=object).columns:

            # Expanding each feature list into columns to carry out PCA
            df_exp = df[column].apply(pd.Series) # rows = cases, columns = values per feature 'colummn'
            #Naming columns with feature name for later reference ('E_0','E_1',...)
            df_exp.columns = [f'{column}'+'_{}'.format(i) for i in range(len(df_exp.columns))]

            # PCA for dimensionality reduction, deciding n_pcs by variance captured, using the expanded df
            pca = PCA(n_components=var_ratio)
            pca_arr = pca.fit_transform(df_exp) #fit PCA and apply reduction
            n_pcs = pca.n_components_ # number of components extracted for the values expanded from feature 'column'

            # Create a df with the principal components extracted as columns - reduced set of components to describe all original values in feature 'column'
            pca_exp_df = pd.DataFrame(pca_arr, index = df.index, columns = [f'{column}' +'_pc{}'.format(i) for i in range(n_pcs)])
            
            # Update the original df, dropping the original feature column before expansion (dtype=object) and replacing with the principal components found
            df = pd.concat([df,pca_exp_df],axis=1).drop(column,axis=1)

            # pca.components has shape [n_components, n_features(columns in df_exp)]. 
            # Each row is a principal component and the value per column is a weight related to the original feature value in df_exp associated with its relevance

            # Finding the biggest weight per component, by going row by row and finding the position of the largest absolute value
            max_weight_per_component = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

            # Most relevant feature value (column in df_exp) per component
            col_feature_per_component = [df_exp.columns[max_weight_per_component[i]] for i in range(n_pcs)]

            for i in range(n_pcs):
                #row to add to pca_df with PCs, dominant features and explained variance
                row_to_add = pd.Series({pca_labels[0] : f'{column}'+'_pc{}'.format(i),
                                        pca_labels[1]: col_feature_per_component[i],
                                        pca_labels[2]: pca.explained_variance_ratio_[i]*100,
                                        pca_labels[3]: pca.components_[i]})
                
                pca_df = pd.concat([pca_df,pd.Series(row_to_add).to_frame().T],ignore_index=True)

            fig = plt.figure(figsize=(8,6))
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Number of Components')
            plt.ylabel('Explained Variance')
            plt.title(f'{column}: [PC={n_pcs}]')
            fig.savefig(os.path.join(self.fig_savepath,f'{self._case}_PCA_{column}'),dpi=200)

        return df, pca_df

class DataPackager(PathConfig):

    def __init__(self,case):
        super().__init__()
        self._case = case
    
    def package_data(self,data_pack,labels):

        #Create the folder to store input datasets
        try:
            os.mkdir(os.path.join(self.input_savepath,self._case))
        except:
            pass
        
        # Storing datasets with corresponding labelss
        for data, label in zip(data_pack,labels):

            with open(os.path.join(self.input_savepath,self._case,f'{label}.pkl'),'wb') as file:
                pickle.dump(data,file)

            print(f'Data packet {label} saved successfully')
        
        # Saving package labels to load later
        with open(os.path.join(self.input_savepath,self._case,'Load_Labels.txt'), 'w') as file:
                for label in labels:
                    file.write(label + '\n')
            

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
    percentage_choice = input('Define the percentage of cases to filter out (default: lower,upper=0,0): ')
    percentages = [float(x) for x in percentage_choice.split(',')]
    X_minmax, y_minmax, X_filtered, y_filtered = dt_processor.filter_minmax([X_df,y_df],bottom=percentages[0],upper=percentages[1])
    
    #Scale input and output features
    scale_choice = input('Select a scaling method (norm/log): ')

    X_scaled = dt_processor.scale_data([X_df.copy(),X_minmax,X_filtered],scaling=scale_choice)
    y_scaled = dt_processor.scale_data([y_df.copy(),y_minmax,y_filtered],scaling=scale_choice)

    dt_processor.plot_scaling(X_df,X_scaled[-1],data_label='inputs')
    dt_processor.plot_scaling(y_df,y_scaled[-1],data_label='outputs')

    # train test splitting
    X_train, X_test, y_train, y_test = train_test_split(X_scaled[-1], y_scaled[-1], test_size=0.25, random_state=2024)
    combine_choice = input('Include the filtered cases into training? (y/n):')
    if combine_choice.lower() == 'y':
        X_train = pd.concat([X_train,X_minmax],axis=0)
        y_train = pd.concat([y_train,y_minmax],axis=0)

    y_test_exp = pd.DataFrame()
    
    # Expand y_test arrays into separate columns for further regression eval
    for column in y_test.select_dtypes(include=object).columns:
        df = y_test[column].apply(pd.Series)
        df.columns = [f'{column}'+'_{}'.format(i) for i in range(len(df.columns))]

        y_test_exp = pd.concat([y_test_exp,df], axis=1)

    pca_choice = input('Carry out dimensionality reduction through PCA? (y/n): ')

    if pca_choice.lower() == 'y':

        # Carry out PCA on scaled outputs for training only
        var_ratio = 0.95
        y_train_reduced, pca_df = dt_processor.PCA_reduction(y_train,var_ratio)

        # Package data for further use training and deploying regression models
        data_pack = [df,X_train,y_train_reduced,X_test,y_test_exp,pca_df]
        labels = ['full','X_train_i','y_train_i_red','X_test_i','y_test_i','PCA_info']

    else:

        # Expand y_train and test columns containing arrays to individual columns per feature value for correct handling by regressor
        y_train_exp = pd.DataFrame()

        # targeting only columns with arrays as dtype=objects
        for column in y_train.select_dtypes(include=object).columns:
            df = y_train[column].apply(pd.Series)
            df.columns = [f'{column}'+'_{}'.format(i) for i in range(len(df.columns))]

            y_train_exp = pd.concat([y_train_exp,df], axis=1)

        # Package data for further use training and deploying regression models
        data_pack = [df,X_train,y_train_exp,X_test,y_test_exp]
        labels = ['full','X_train_i','y_train_i','X_test_i','y_test_i']
    
    dt_packager.package_data(data_pack,labels)



if __name__ == "__main__":
    main()