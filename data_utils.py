##########################################################################
#### Data load, pre-process and package utilities
#### Author : Juan Pablo Valdes and Fuyue Liang
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import pandas as pd
import ast
import os
import pickle
from paths import PathConfig
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,RobustScaler,PowerTransformer,QuantileTransformer


COLOR_MAP = cm.get_cmap('viridis', 30)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Computer Modern']})

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 15
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE + 2)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

############################ LOG SCALER ########################################

class LogScaler:
    def __init__(self,base=10):
        self.base = base
        self.log_base = np.log(self.base)

    def fit(self, X, y=None):
        # No operation needed during fitting
        return self
    
    def transform(self,X):
        # use the log modulus transformation: preserves zero and the function acts like the log (base 10) function when non-zero
        log_modulus_X = np.sign(X) * np.log(np.abs(X)+1) / self.log_base
        return log_modulus_X
    
    def inverse_transform(self, X):
        # inverse scaling using the logarithm with a specified base
        inverse_X = X * self.log_base
        # invserse operation to compute absolute value and sign
        sign_X = np.sign(inverse_X)
        abs_log_X = np.exp(np.abs(inverse_X)) - 1
        # inverse operation to compute the original value
        inverse_transformed_X = sign_X * abs_log_X
        return inverse_transformed_X
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)

############################ DATA READER ########################################

class DataReader(PathConfig):

    def __init__(self,case, data):
        super().__init__()
        self._case = case
        self._data = data

    def collect_csv_pkl(self):
        
        csv_list = []
        DOE_list = []

        # data and DOE paths for the case under consideration
        csv_path = os.path.join(self.raw_datapath,self._case, self._data) #output data
        doe_path = os.path.join(self.label_datapath,self._case,self._data) #input data/labels

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

############################ DATA PROCESSOR ########################################

class DataProcessor(PathConfig):

    def __init__(self,case,data):
        super().__init__()
        self._case = case
        self._data = data

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
                    # Convert flat index inside array into a row,column position - row=case
                    the_index = np.unravel_index(np.argmin(stacked_arr),stacked_arr.shape)

                elif where == 'max':
                    the_index = np.unravel_index(np.argmax(stacked_arr),stacked_arr.shape)
                
                index_in_df = df_column.index[the_index[0]] #case based on row with min/max value
                
                # drop the case found and search for next min/max at loop restart
                df_column = df_column.drop(index_in_df)

                indices_in_df.append(index_in_df)

        # If feature is not a collection of arrays
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

        #search minmax values per feature
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

    def apply_scaling(self, scaler, data_pack: list) -> list:
        
        scaled_data = []
        # laying out the input datapacks to further update with their scaled values
        scaled_datapacks = {idx: dataframe.copy() for idx,dataframe in enumerate(data_pack)}

        # scale each feature
        for column in data_pack[0].columns:

            # scale selected feature in each data-pack
            for df_idx, df in enumerate(data_pack):
                # get the scaler for the entire dataset
                if df_idx == 0:
                    # Apply scaler, reshaping into a column vector (n,1) for scaler to work if output feature is an array
                    if df[column].dtype == 'object':
                        # check if the data_name is ini, yes then need to fit the scaler, otherwise load the scaler from ini
                        if self._data != 'ini':
                            with open(os.path.join(self.input_savepath,self._case, 'ini', f'scaler_y_{column.split()[0]}.pkl'),'rb') as f:
                                scaler = pickle.load(f)
                        else:
                            flat_list = [ele_val for ele in df[column] for ele_val in ele]
                            flat_arr = np.array(flat_list).reshape(-1,1)
                            scaler.fit(flat_arr)
                            # save the scaler for later augmentation
                            with open(os.path.join(self.input_savepath,self._case, self._data, f'scaler_y_{column.split()[0]}.pkl'),'wb') as f:
                                pickle.dump(scaler,f)

                        df[column] = df[column].apply(lambda x: scaler.transform(x.reshape(-1,1)))            
                        # reshaping back to a 1D list
                        df[column] = df[column].apply(lambda x: x.reshape(-1,))
                    else:
                        # check if the data_name is ini, yes then need to fit the scaler, otherwise load the scaler from ini
                        if self._data != 'ini':
                            with open(os.path.join(self.input_savepath,self._case, 'ini', f'scaler_X_{column.split()[0]}.pkl'),'rb') as f:
                                scaler = pickle.load(f)
                        # Save the scaler for later inverse transformation for input
                        else:
                            scaler.fit(df[column].values.reshape(-1,1))
                            with open(os.path.join(self.input_savepath,self._case, self._data, f'scaler_X_{column.split()[0]}.pkl'),'wb') as f:
                                pickle.dump(scaler,f)
                        df[column] = scaler.transform(df[column].values.reshape(-1,1))
                        df[column] = df[column].values.reshape(-1,)
                
                # We dont fit (only transform) datapacks 1,2 again to maintain consistency with the entire set scaling executed above
                else:
                    if df[column].dtype == 'object':
                        df[column] = df[column].apply(lambda x: scaler.transform(x.reshape(-1,1)))            
                        # reshaping back to a 1D list
                        df[column] = df[column].apply(lambda x: x.reshape(-1,))
                    else:
                        df[column] = scaler.transform(df[column].values.reshape(-1,1))
                        df[column] = df[column].values.reshape(-1,)

                #append scaled feature to each corresponding datapack
                scaled_datapacks[df_idx][column] = df[column].copy()

        scaled_data = [scaled_datapacks[idx] for idx in range(len(scaled_datapacks))]

        return scaled_data

    def scale_data(self,data_pack: list, scaling: str) -> list:
        '''
        To use this function:
        data_pack is supposed to contain one entire dataset and subsets of the entire dataset,
        e.g., data_pack = [entire (X/y), minmax_values_pack, filtered_packs]
        '''

        scalers = {'log': LogScaler(base=10),
                   'robust': RobustScaler(with_centering=False, with_scaling=True, quantile_range=(25.0,75.0)),
                   'power': PowerTransformer(),
                   'quantile': QuantileTransformer(output_distribution='normal')}
        
        # create scaler only for ini
        if self._data == 'ini':
            # create a pipeline with the scaler and/or MinMaxScaler
            if scaling != 'norm':
                scaling_pipeline = Pipeline([
                    (scaling, scalers.get(scaling)),
                    ('minmax', MinMaxScaler(feature_range=(-1,1)))
                ])
            else:
                scaling_pipeline = Pipeline([
                    ('minmax', MinMaxScaler(feature_range=(-1,1)))
                ])
            
            scaler = scaling_pipeline
        else:
            scaler = None

        mmscaled_data = self.apply_scaling(scaler, data_pack)
 
        return mmscaled_data
    
    # Visualize data before and after scaling
    def plot_scaling(self, original_data, scaled_data, scaled_extreme_cases, data_label):
        
        num_features = len(original_data.columns)
        
        fig,ax = plt.subplots(num_features,3, figsize=(18,int(num_features)*5))
        plt.subplots_adjust(hspace=0.8)

        # If only one output feature is selected, reshape the axes to allow plotting
        if num_features == 1:
            ax = ax.reshape((1, 3))
        
        reset_data = original_data.reset_index(drop=True)
        reset_scaled = scaled_data.reset_index(drop=True)
        reset_extreme = scaled_extreme_cases.reset_index(drop=True)

        for i, column in enumerate(reset_data.columns):

            # Instructions for output features read as arrays from csv
            if reset_data[column].dtype == 'object':

                for j in reset_data.index:
                    ax[i,0].plot(reset_data[column][j])
                    ax[i,0].set_title(f'Data before: {column}')
                for k in reset_scaled.index:
                    ax[i,1].plot(reset_scaled[column][k])
                    ax[i,1].set_title(f'Data after: {column}')
                for l in reset_extreme.index:
                    ax[i,2].plot(reset_extreme[column][l])
                    ax[i,2].set_title(f'Scaled Data from extreme cases: {column}')
            # Scalar input features
            else:
                ax[i,0].plot(reset_data[column])
                ax[i,0].set_title(f'Data before: {column}')
                ax[i,1].plot(reset_scaled[column])
                ax[i,1].set_title(f'Data after: {column}')
                ax[i,2].plot(reset_extreme[column])
                ax[i,2].set_title(f'Scaled Data from extreme cases: {column}')

        fig.savefig(os.path.join(self.fig_savepath, self._case, self._data,f'{data_label}'),dpi=200)
        plt.show()
    
    def PCA_reduction(self,df,var_ratio, datasample: str):
        '''
        Perform PCA reduction and plot diagrams of Explained variance vs number of principal components
        '''
        # Empty Dataframe for PCs results and principal axes
        pca_labels = ['PCs', 'Dominant_Features', 'Explained_Var']
        pca_info_df = pd.DataFrame(columns=pca_labels)

        fig,ax = plt.subplots(figsize=(8,6))
        colors = sns.color_palette('muted',len(df.columns))
        # Carry out expansion and PCA per array features(exclude scalar outputs if any)
        for idx, column in enumerate(df.select_dtypes(include=object).columns):

            # Expanding each feature list into columns to carry out PCA
            df_exp = df[column].apply(pd.Series) # rows = cases, columns = values per feature 'colummn'
            # Naming columns with feature name for later reference ('E_0','E_1',...)
            df_exp.columns = [f'{column}'+'_{}'.format(i) for i in range(len(df_exp.columns))]

            # PCA for dimensionality reduction, deciding n_pcs by variance captured, using the expanded df
            pca = PCA(n_components=var_ratio)
            pca_arr = pca.fit_transform(df_exp) #fit PCA and apply reduction
            # Save the transform for later (reverse pca)
            if not os.path.exists(os.path.join(self.pca_savepath, self._case, datasample)):
                os.makedirs(os.path.join(self.pca_savepath, self._case, datasample))
            with open(os.path.join(self.pca_savepath, self._case, datasample, f'pca_model_{column}.pkl'), 'wb') as f:
                pickle.dump(pca,f)
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
                #row to add to pca_info_df with PCs, dominant features and explained variance
                row_to_add = pd.Series({pca_labels[0] : f'{column}'+'_pc{}'.format(i),
                                        pca_labels[1]: col_feature_per_component[i],
                                        pca_labels[2]: pca.explained_variance_ratio_[i]*100})
                
                pca_info_df = pd.concat([pca_info_df,pd.Series(row_to_add).to_frame().T],ignore_index=True)

            
            plt.plot(np.cumsum(pca.explained_variance_ratio_)*100, linewidth=2.5,color=colors[idx], label=f'{column}: [PC={n_pcs}]')
            
        plt.xlabel('Number of Components',fontweight='bold',fontsize=30)
        plt.ylabel(r'Explained Variance [\%]',fontweight='bold',fontsize=30)
        plt.tick_params(axis='both',labelsize=20)
        # plt.title(f'{column}: [PC={n_pcs}]')
        plt.legend(#title=f'No. of ', title_fontsize=25,
                                loc='lower right',fontsize=18,
                                edgecolor='black', frameon=True)
        plt.axhline(y=95, color='black', linestyle='--', linewidth=2)
        plt.text(0.02, 90, r'Exp. Var. = 95\%', transform=ax.get_yaxis_transform(), fontsize=18)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        fig.savefig(os.path.join(self.fig_savepath,f'{self._case}_PCA.png'),dpi=200)

        return df, pca_info_df

############################ DATA PACKAGER ########################################

class DataPackager(PathConfig):

    def __init__(self,case,data):
        super().__init__()
        self._case = case
        self._data = data

    def expand_targets(self, target_df: pd.DataFrame) -> pd.DataFrame:
        
        expanded_df = pd.DataFrame()
    
        # Expand y_test arrays into separate columns for further regression eval
        for column in target_df.select_dtypes(include=object).columns:
            df = target_df[column].apply(pd.Series)
            df.columns = [f'{column}'+'_{}'.format(i) for i in range(len(df.columns))]

            expanded_df = pd.concat([expanded_df,df], axis=1)
        
        return expanded_df
    
    def package_data(self,data_pack,labels,datasample: str):

        if datasample == 'ini':
            data_dir = os.path.join(self.input_savepath,self._case, datasample)
        else:
            data_dir = os.path.join(self.resample_savepath,self._case, datasample)
        
        #Create the folder to store input datasets
        try:
            os.mkdir(data_dir)
        except:
            pass
        
        # Storing datasets with corresponding labels
        for data, label in zip(data_pack,labels):

            with open(os.path.join(data_dir,f'{label}.pkl'),'wb') as file:
                pickle.dump(data,file)

            print(f'Data packet {label} saved successfully')
        
        # Saving package labels to load later
        with open(os.path.join(data_dir,'Load_Labels.txt'), 'w') as file:
                for label in labels:
                    file.write(label + '\n')

############################ DATA LOADER ########################################

class DataLoader(PathConfig):

    def __init__(self,case):

        super().__init__()
        self._pca = False
        self.case = case
    
    @property
    def pca(self):
        return self._pca
    
    def check_pca(self,label_package):
        # Checking in PCA has been applied to the dataset
        if 'PCA_info' in label_package:
            self._pca = True
        else:
            self._pca = False

    def load_packs(self,dir):
        
        label_package = []
        data_packs = []

        labelfile_dir = os.path.join(dir,'Load_Labels.txt')
        
        # Read package names generated from Load_Label.txt to import accordingly
        with open(labelfile_dir, 'r') as file:
            lines = file.readlines()

            for line in lines:
                label_package.append(line.split('\n')[0])

        # change or keep the property pca
        self.check_pca(label_package)
        
        # Save only train, test packs
        label_package = [item for item in label_package if item not in ['full', 'PCA_info']]
        
        # Load pickle files
        for label in label_package:

            data_path = os.path.join(dir,f'{label}.pkl')

            if os.path.exists(data_path):

                data_pack = pd.read_pickle(data_path)          
                data_packs.append(data_pack)
        
        return data_packs
    
    def select_augmentdata(self,df,num_cases:int):

        # shuffle df for randomisation
        # df_shuffled = df.sample(frac=1,random_state=2024)
        df_selected = df.iloc[:num_cases]
        
        return df_selected

    def augment_data(self,data_sample : str):
        
        data_aug_dir = os.path.join(self.resample_savepath, self.case, data_sample)
        ini_data_dir = os.path.join(self.input_savepath, self.case, 'ini')

        aug_packs = self.load_packs(data_aug_dir)
        ini_packs = self.load_packs(ini_data_dir)

        print('-'*72)
        # print(f'Augmenting {aug_packs[0].shape[0]} cases from sample {data_sample} to initial training data with initial size: {ini_packs[0].shape[0]}')
        print(f'Number of cases from sample {data_sample}: {aug_packs[0].shape[0]}.')
        print('-'*72)

        # provide the number of cases to be augmented
        num_rows = input('Provide the number of cases to be augmented or all:')

        if num_rows.lower() == 'all':
            # Augment training data with all loaded sampled data
            # data_packs[0] = X_train, data_packs[1] = y_train(ready as model input),  data_pack[-1] = Y_TRAIN_RAW
            X_train_aug = pd.concat([ini_packs[0],aug_packs[0]], ignore_index= True)
            y_train_aug_raw = pd.concat([ini_packs[-1],aug_packs[-1]], ignore_index= True)
        else:
            num_cases = int(num_rows)
            # select the cases depending on the num_cases
            X_selected_aug = self.select_augmentdata(aug_packs[0],num_cases)
            y_selected_aug = self.select_augmentdata(aug_packs[-1],num_cases)

            # Augment training data with selected samples
            X_train_aug = pd.concat([ini_packs[0],X_selected_aug], ignore_index= True)
            y_train_aug_raw = pd.concat([ini_packs[-1],y_selected_aug], ignore_index= True)

        # perform either expand_target or PCA_reduction here
        dt_processor = DataProcessor(self.case, data_sample)
        dt_packager = DataPackager(self.case, data_sample)

        if self._pca:
            # Carry out PCA on scaled outputs for training only
            var_ratio = 0.95
            y_train_aug, _ = dt_processor.PCA_reduction(y_train_aug_raw,var_ratio, datasample=data_sample)

        else:
            y_train_aug = dt_packager.expand_targets(y_train_aug_raw)

        # Returning the newly augmented training sets and the original testing sets split in input.py
        data_packs = [X_train_aug, y_train_aug,ini_packs[2],ini_packs[3]]

        return data_packs