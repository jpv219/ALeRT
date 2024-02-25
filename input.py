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

    def __init__(self):
        super().__init__()

    def collect_csv_pkl(self,case):
        
        csv_list = []
        DOE_list = []
        run_list = []
        run_count = 1 # Assuming at least one run exists

        # data and DOE paths for the case under consideration
        csv_path = os.path.join(self.raw_datapath,case) #output data
        doe_path = os.path.join(self.label_datapath,case) #input data/labels

        file_count = len(os.listdir(csv_path))

        # Loop concatenating DOE and csv files from different parametric runs
        for i in range(1,file_count+1):

            csv_filename = f'{case}_{i}.csv'
            doe_filename = f'LHS_{case}_{i}.pkl'
            
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
        for i in range(1,len(data_df['Run_ID'])):
            
            if data_df['Run_ID'].iloc[i-1] == data_df['Run_ID'].iloc[i]:
                continue
            else:
                run_count +=1
                #Extract Run_ID once a new Run is identified
                run_list.append(data_df['Run_ID'].iloc[i-1])

        run_list.append(data_df['Run_ID'].iloc[-1])

        return data_df, DOE_df, run_list

    def combine_data(self,case):
        
        data_df, DOE_df, run_list = self.collect_csv_pkl(case)
        
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
        data_df_reshaped = data_df_sorted.groupby('index').agg(lambda x: x.tolist())

        # Merge input parameters (labels) with hydrodynamic data output (target)
        df = pd.concat([df_DOE_filtered,data_df_reshaped],axis=1)

        return df


class DataProcessor(PathConfig):

    def __init__(self):
        super().__init__()

class DataPackager(PathConfig):

    def __init__(self):
        super().__init__()

    def scale_df(self):
        pass

def main():
    
    case_name = input('Select a study to process raw datasets (sp_geom, surf, geom): ')

    # Datareader instance
    dt_reader = DataReader()

    #Combine csv and DOE label files
    df = dt_reader.combine_data(case_name)



if __name__ == "__main__":
    main()