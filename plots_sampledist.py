##########################################################################
#### Plottings for thesis
#### Author : Fuyue Liang
### First commit: Jun 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import pandas as pd
import os
import numpy as np
from paths import PathConfig
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.weight": "bold",
    "font.serif": ['Computer Modern']})

#label_list = {'Impeller_Diameter (m)': r'Impeller diameter', 'Frequency (1/s)':r'Frequency', 'Clearance (m)':r'Clearance', 'Blade_width (m)':r'Blade width',
#             'Blade_thickness (m)': r'Blade thickness', 'Nblades': r'Blade number', 'Inclination': r'Inclination'}
label_list = {'Bar_Width (mm)': r'$W$', 'Bar_Thickness (mm)' : r'$Th$', 'Radius (mm)': r'$R_p$', 'Nbars' : r'$n_{cr}$',
        'Flowrate (m3/s)': r'$Q_c$', 'Angle': r'$\theta$', 'NElements': r'$n_{E}$','Re':r'$Re$'}
data_label = {'random':r'Random', 'dt':r'DT-guided','gsx':r'GSx'}

class SpacePlotter(PathConfig):

    def __init__(self,case,data_type):
        super().__init__()

        self.case = case
        self.data = data_type

    def define_random_center(self, in_idx_list,ini_df):
        centroid = np.array([0,0,0])
        ini_df_sub = ini_df[ini_df.columns[in_idx_list]]
        distances = [np.linalg.norm(ini_df_sub.loc[idx].to_numpy()-centroid) for idx in ini_df_sub.index]
        point_idx = ini_df_sub.index[np.argmin(distances)]
        the_point = ini_df_sub.loc[point_idx]
        print(f'Center point idx: {point_idx}')
        
        fig1 = plt.figure(figsize = (11, 10))

        col1 = ini_df.columns[in_idx_list[0]]
        col2 = ini_df.columns[in_idx_list[1]]
        col3 = ini_df.columns[in_idx_list[2]]

        ref_x1 = ini_df[col1].to_numpy()
        ref_x2 = ini_df[col2].to_numpy()
        ref_x3 = ini_df[col3].to_numpy()

        c_x1 = the_point[col1]
        c_x2 = the_point[col2]
        c_x3 = the_point[col3]

        ax = plt.axes(projection ="3d")
        # Creating plot
        ax.scatter3D(ref_x1, ref_x2, ref_x3, s=100)
        ax.scatter3D(c_x1, c_x2, c_x3, s=100, c='red', label='Point')
        fig1.savefig(f'/home/jpv219/Documents/ML/ALeRT/figs/sp_geom/input_space/sampledist_random_center.png',dpi=300)
        plt.show()

    def plot_point_dist3d(self, in_idx_list,iter_list,ini_df,df):
        
        col1 = df.columns[in_idx_list[0]]
        col2 = df.columns[in_idx_list[1]]
        col3 = df.columns[in_idx_list[2]]

        ref_x1 = ini_df[col1].to_numpy()
        ref_x2 = ini_df[col2].to_numpy()
        ref_x3 = ini_df[col3].to_numpy()
        x1 = df[col1].to_numpy()
        x2 = df[col2].to_numpy()
        x3 = df[col3].to_numpy()
        
        for iter in iter_list:
            # plot the 3d scatter for initial data set
            fig1 = plt.figure(figsize = (11, 10))
            ax = plt.axes(projection ="3d")
            # Creating plot
            ax.scatter3D(ref_x1, ref_x2, ref_x3, s=60, color = "#FFAAAA80",label=r'Initial')
            ax.scatter3D(x1[:iter], x2[:iter], x3[:iter], marker='^', s=200, color = "blue",label=f'{data_label.get(self.data)}')

            # figure format
            ax.set_xlabel(f'{label_list.get(col1,col1)}', fontsize =35)
            ax.set_ylabel(f'{label_list.get(col2,col2)}', fontsize =35)
            ax.set_zlabel(f'{label_list.get(col3,col3)}', fontsize =35)

            ax.set_xticks([-1,-0.5,0,0.5,1])
            ax.set_yticks([-1,-0.5,0,0.5,1])
            ax.set_zticks([-1,-0.5,0,0.5,1])
            ax.tick_params(labelsize=25)
            ax.xaxis.labelpad=30
            ax.yaxis.labelpad=30
            ax.zaxis.labelpad=20
            ax.axes.set_xlim3d(left=-1, right=1) 
            ax.axes.set_ylim3d(bottom=-0.98, top=0.98) 
            ax.axes.set_zlim3d(bottom=-0.98, top=0.98)
            ax.legend(loc='upper right',fontsize=30)
            
            fig1.tight_layout()
            fig1.savefig(os.path.join(self.fig_savepath, self.case, f'input_space/sampledist_{self.data}_iter{iter}.png'),dpi=300)
            plt.show()

    def plot_point_dist2d(self, in_idx_list,ini_df,df):
        
        col1 = df.columns[in_idx_list[0]]
        col2 = df.columns[in_idx_list[1]]

        ref_x1 = ini_df[col1].to_numpy()
        ref_x2 = ini_df[col2].to_numpy()
        
        x1 = df[col1].to_numpy()
        x2 = df[col2].to_numpy()

        # plot the 3d scatter for initial data set
        fig1 = plt.figure(figsize = (8, 6))
        
        # Creating plot
        plt.scatter(ref_x1, ref_x2, s=30, color = "red",label=r'initial')
        plt.scatter(x1, x2, marker='^', s=30, color = "blue",label=f'{self.data}')

        # figure format
        plt.xlabel(f'{label_list.get(col1,col1)}', fontsize =25)
        plt.ylabel(f'{label_list.get(col2,col2)}', fontsize =25)

        plt.xticks([-1,-0.5,0,0.5,1])
        plt.yticks([-1,-0.5,0,0.5,1])
        plt.tick_params(labelsize=18)
        plt.legend(loc='upper right',fontsize=20)
        
        fig1.tight_layout()
        fig1.savefig(os.path.join(self.fig_savepath, self.case, f'input_space/sampledist_{self.data}_{label_list.get(col1,col1)}{label_list.get(col2,col2)}.png'),dpi=200)
        plt.show()

    def plot_joint_3dscatter(self, in_idx_list,iter_list,ini_df,dfs,data_names):
        
        colors = {'random':"blue", 'dt':"orange",'gsx':"green"}

        markers = {'random': 'x', 'dt': 'v', 'gsx': 's'}

        # Initialize the figure and 3D axis
        fig1 = plt.figure(figsize=(11, 10))
        ax = plt.axes(projection="3d")
        
        for df,data_name in zip(dfs,data_names):
        
            col1 = df.columns[in_idx_list[0]]
            col2 = df.columns[in_idx_list[1]]
            col3 = df.columns[in_idx_list[2]]

            x1 = df[col1].to_numpy()
            x2 = df[col2].to_numpy()
            x3 = df[col3].to_numpy()
        
            for iter in iter_list:

                ax.scatter3D(x1[:iter], x2[:iter], x3[:iter], marker=markers.get(data_name), s=200, color = colors.get(data_name),edgecolors = 'k',label=f'{data_label.get(data_name,data_name)}')

        # figure format
                
        ref_x1 = ini_df[col1].to_numpy()
        ref_x2 = ini_df[col2].to_numpy()
        ref_x3 = ini_df[col3].to_numpy()
        ax.scatter3D(ref_x1, ref_x2, ref_x3, s=60, color = "#FFAAAA80",label=r'Initial')
        ax.set_xlabel(f'{label_list.get(col1,col1)}', fontsize =35)
        ax.set_ylabel(f'{label_list.get(col2,col2)}', fontsize =35)
        ax.set_zlabel(f'{label_list.get(col3,col3)}', fontsize =35)

        ax.set_xticks([-1,-0.5,0,0.5,1])
        ax.set_yticks([-1,-0.5,0,0.5,1])
        ax.set_zticks([-1,-0.5,0,0.5,1])
        ax.tick_params(labelsize=25)
        ax.xaxis.labelpad=30
        ax.yaxis.labelpad=30
        ax.zaxis.labelpad=20
        ax.axes.set_xlim3d(left=-1, right=1) 
        ax.axes.set_ylim3d(bottom=-0.98, top=0.98) 
        ax.axes.set_zlim3d(bottom=-0.98, top=0.98)
        ax.legend(loc='upper left',fontsize=22)
        
        fig1.tight_layout()
        fig1.savefig(os.path.join(self.fig_savepath, self.case, f'input_space/sampledist_combined_iter{iter}.png'),dpi=300)
        plt.show()

def main():

    case_name = input('Select a study to process raw datasets (sp_(sv)geom): ')

    data_name = input('Select a dataset to process from the study selected above (random, dt, gsx): ')

    iter = input('Provide the sample numbers in 3 iterations (e.g. 10,20,30): ')
    iter_list = [int(x) for x in iter.split(',')]

    plotter = SpacePlotter(case_name,data_name)

    ini_pkl = os.path.join(plotter.input_savepath, case_name, 'ini','X_train_i.pkl')
    random_pkl = os.path.join(plotter.resample_savepath, case_name,'random', f'X_random.pkl')

    if data_name == 'random':
        input_pkl = os.path.join(plotter.resample_savepath, case_name,data_name, f'X_{data_name}.pkl')
    else:
        input_pkl = os.path.join(plotter.resample_savepath, case_name,data_name, f'X_train_{data_name}.pkl')

    ini_df = pd.read_pickle(ini_pkl)
    random_df = pd.read_pickle(random_pkl)
    input_df = pd.read_pickle(input_pkl)

    params = input_df.columns
    param_idx = [(idx,param) for idx,param in enumerate(params)]

    print(param_idx)

    in_idx = input('Select 2 or 3 input parameters idx for plotting (separated by ,) : ')

    in_idx_list = [int(x) for x in in_idx.split(',')]

    if len(in_idx_list) == 3:
        
        plot_dimen = input('Select 2D or 3D plotting:')
        if plot_dimen.lower() == '2d':
            in_idx_list_sub1 = in_idx_list[:2]
            plotter.plot_point_dist2d(in_idx_list_sub1,ini_df,input_df)
            in_idx_list_sub2 = [in_idx_list[0], in_idx_list[2]]
            plotter.plot_point_dist2d(in_idx_list_sub2,ini_df,input_df)
            in_idx_list_sub3 = [in_idx_list[1], in_idx_list[2]]
            plotter.plot_point_dist2d(in_idx_list_sub3,ini_df,input_df)
        else:
            # define_random_center(in_idx_list,random_df) #35
            plotter.plot_point_dist3d(in_idx_list,iter_list,ini_df, input_df)
    
    elif len(in_idx_list) == 2:

        plotter.plot_point_dist2d(in_idx_list,ini_df,input_df)

    combined = input('Plot all together?(y/n): ')

    if combined.lower() == 'y':

        dt_pkl = os.path.join(plotter.resample_savepath, case_name,'dt', 'X_train_dt.pkl')
        dt_df = pd.read_pickle(dt_pkl)
        gsx_pkl = os.path.join(plotter.resample_savepath, case_name,'gsx', 'X_train_gsx.pkl')
        gsx_df = pd.read_pickle(gsx_pkl)

        data_names = ['random','dt','gsx']

        dfs = [random_df,dt_df,gsx_df]

        plotter.plot_joint_3dscatter(in_idx_list, iter_list, ini_df, dfs, data_names)

if __name__ == "__main__":
    main()
