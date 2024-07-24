####################################################
#### New GSx sampling for thesis
##### Run the smpling.py for gsx before this scipts
####################################################

import pandas as pd
import os

gsx_path = '/home/jpv219/Documents/ML/ALeRT/resample/sp_geom/gsx'
dfgsx_path = '/home/jpv219/Documents/ML/ALeRT/resample/sp_geom/gsx/trial1/'
random_path = '/home/jpv219/Documents/ML/ALeRT//resample/sp_geom/random/'

dfgsx = pd.read_pickle(os.path.join(dfgsx_path,'gsx_df.pkl'))
dfx = pd.read_pickle(os.path.join(random_path,'X_random.pkl'))
dfy = pd.read_pickle(os.path.join(random_path,'y_random_raw.pkl'))

dfxgsx = dfx.loc[dfgsx.index]
dfygsx = dfy.loc[dfgsx.index]

dfxgsx.to_pickle(os.path.join(gsx_path,'X_train_gsx.pkl'))
dfygsx.to_pickle(os.path.join(gsx_path,'y_train_gsx_raw.pkl'))

labels = ['X_train_gsx','y_train_gsx_raw']

# Saving package labels to load later
with open(os.path.join(gsx_path,'Load_Labels.txt'), 'w') as file:
        for label in labels:
            file.write(label + '\n')

print('Work done')
