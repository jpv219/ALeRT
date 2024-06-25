##########################################################################
#### Plottings for thesis
#### Author : Fuyue Liang
### First commit: Jun 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import pandas as pd
import os
from paths import PathConfig
from model_utils import ModelEvaluator
from model_lib import ModelConfig
from data_utils import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Computer Modern']})


def main():

    case_name = input('Select a study to process raw datasets (sp_(sv)geom): ')

    model_choice = input('Select a regressor to train and deploy (dt, xgb, rf, svm, knn, mlp_br, mlp): ')
    
    # Global paths configuration
    PATH = PathConfig()

    # Empty dataframe for storing
    df_savepath = os.path.join(PATH.fig_savepath,case_name,f'{model_choice}_perf.csv')
    
    if not os.path.exists(df_savepath):
        print('-'*72)
        print('No CSV is saved for plotting model performance, initial model is about to be trained....')
        print('-'*72)
    

        df = pd.DataFrame()

        #################################################
        # Train the initial model with ini data set
        #################################################
        dataloader = DataLoader(case_name)
        
        inidata_dir = os.path.join(PATH.input_savepath, case_name, 'ini')

        # load initial data packs and retrieve whether pca has been executed
        data_packs = dataloader.load_packs(inidata_dir)
        case_num = len(data_packs[0])
        pca = dataloader.pca

        # Model configurer
        m_config = ModelConfig()
        
        # selecting corresponding wrapper, hyperparams and model_name
        wrapper_model = m_config.get_wrapper(model_choice)
        model_params = m_config.get_hyperparameters(model_choice)
        model_name = m_config.get_model_name(model_choice)

        # Add input_size, output_size and n_features for MLP models and negate early kfold
        if model_choice in ['mlp', 'mlp_br']:
            model_params['input_size'] = data_packs[0].shape[-1]
            model_params['output_size'] = data_packs[1].shape[-1]
            is_mlp = True

            # Count the number of individual features in the input data
            features = data_packs[1].columns
            unique_features = set()

            for feat in features:
                # split the column number from the feature name
                name = feat.rsplit('_',1)[0]
                unique_features.add(name)

            n_features = len(unique_features)
            model_params['n_features'] = n_features

            do_kfold = 'n'
            ksens = 'n'
        
        # only ask for early kfold on sklearn native models
        else:

            do_kfold = input('Perform pre-Kfold cross validation? (y/n): ')
        
            # Decide whether to do pre-kfold and include k sensitivity
            if do_kfold.lower() == 'y':
                ksens = input('Include K-sensitivity? (y/n): ')
            else:
                ksens = 'n'

            is_mlp = False
        
        do_hp_tune = input('Perform hyperparameter tuning cross-validation? (y/n): ')

        cv_options = {'do_kfold': True if do_kfold.lower() == 'y' else False,
            'ksens' : True if ksens.lower() == 'y' else False,
            'do_hp_tune': True if do_hp_tune.lower() == 'y' else False}

        # Instantiating the wrapper with the corresponding hyperparams
        model_instance = wrapper_model(**model_params)

        # Getting regressor object from wrapper
        model = model_instance.init_model()

        # Regression training and evaluation
        tuned_model = model_instance.model_train(data_packs, model, 
                                            cv_options, model_name)
        
        print('-'*72)
        print('-'*72)
        print(f'Saving {model_name} best model...')
        
        # Save best perfoming trained model based on model_train cross_validation filters and steps selected by the user
        best_model_path = os.path.join(PATH.bestmodel_savepath, model_name)
        # Check if the subfolder for the model exists
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        model_instance.save_model(tuned_model,best_model_path,is_mlp)

        model_eval = ModelEvaluator(tuned_model, model_name, data_packs,case_name,pca, datasample='ini')

        y_pred = model_eval.predict(model_eval.X_test_df, model_eval.y_test_df)

        r2_ini = r2_score(model_eval.y_test, y_pred)
        mse_ini = mean_squared_error(model_eval.y_test, y_pred)
        mae_ini = mean_absolute_error(model_eval.y_test, y_pred)

        df[f'r2_ini_{case_num}'] = [r2_ini]
        df[f'mse_ini_{case_num}'] = [mse_ini]
        df[f'mae_ini_{case_num}'] = [mae_ini]
        df.to_csv(df_savepath,index=False)
        print('-'*72)
        print('csv saved with performance based on ini.')
        print('-'*72)

        #################################################
        # Train the aug model with aug data set
        #################################################
        # augment dataset with selected sample data
        data_choice_list = ['random', 'dt', 'gsx']

        for data_choice in data_choice_list:
            for idx in range(3):
                # Load best trained model
                print('-'*72)
                print(f'This is Iteration {idx}.')
                data_packs = dataloader.augment_data(data_choice)
                case_num = len(data_packs[0])

                if model_choice in ['mlp', 'mlp_br']:
                    is_mlp = True
                    best_model_path = os.path.join(PATH.bestmodel_savepath, model_name,'best_model.keras')

                    model_params['input_size'] = data_packs[0].shape[-1]
                    model_params['output_size'] = data_packs[1].shape[-1]

                    # Count the number of individual features in the input data
                    features = data_packs[1].columns
                    unique_features = set()

                    for feat in features:
                        # split the column number from the feature name
                        name = feat.rsplit('_',1)[0]
                        unique_features.add(name)

                    n_features = len(unique_features)
                    model_params['n_features'] = n_features

                else:
                    is_mlp = False
                    best_model_path = os.path.join(PATH.bestmodel_savepath, model_name,'best_model.pkl')

                # Instantiating the wrapper with the corresponding hyperparams
                model_instance = wrapper_model(**model_params)

                best_model = model_instance.load_model(best_model_path, is_mlp)

                # Train model with new data, w/o further tuning or cross validation
                cv_options = {'do_kfold': False,
                    'ksens' : False,
                    'do_hp_tune': False}
                
                re_trained_model = model_instance.model_train(data_packs, best_model,
                                        cv_options, model_name)
                
                model_eval = ModelEvaluator(re_trained_model, model_name, data_packs,case_name,pca, datasample=data_choice)

                y_pred = model_eval.predict(model_eval.X_test_df, model_eval.y_test_df)

                r2 = r2_score(model_eval.y_test, y_pred)
                mse = mean_squared_error(model_eval.y_test, y_pred)
                mae = mean_absolute_error(model_eval.y_test, y_pred)

                df[f'r2_{data_choice}_{case_num}'] = [r2]
                df[f'mse_{data_choice}_{case_num}'] = [mse]
                df[f'mae_{data_choice}_{case_num}'] = [mae]
                df.to_csv(df_savepath,index=False)
                print('-'*72)
                print(f'csv saved with performance based on {data_choice} of cases {case_num}.')
                print('-'*72)
    else:
        print('-'*72)
        print(f'{model_choice} CSV is saved. Start plotting...')
        print('-'*72)

        df = pd.read_csv(df_savepath)

        # Load the best models
        if model_choice in ['mlp', 'mlp_br']:
                    is_mlp = True
                    best_model_path = os.path.join(PATH.bestmodel_savepath, model_name,'best_model.keras')

        else:
            is_mlp = False
            best_model_path = os.path.join(PATH.bestmodel_savepath, model_name,'best_model.pkl')

        # Instantiating the wrapper with the corresponding hyperparams
        model_instance = wrapper_model(**model_params)

        best_model = model_instance.load_model(best_model_path, is_mlp)

        dataloader = DataLoader(case_name)
    
        data_choice_list = ['ini','random', 'dt', 'gsx']
        
        # evaluation and plotting
        for data_choice in data_choice_list:
            if data_choice == 'ini':
                inidata_dir = os.path.join(PATH.input_savepath, case_name, 'ini')
                data_packs = dataloader.load_packs(inidata_dir)
            else:
                data_packs = dataloader.augment_data(data_choice)
            pca = dataloader.pca
            
            model_instance.model_evaluate(best_model, model_name, data_packs,case_name,pca,datasample = data_choice)

    #################################################
    # Plot the performance with scatter plot
    #################################################
    
    metrics_list = ['r2', 'mae', 'mse']
    label_list = {'r2': r'$R^2$', 'mae':r'MAE', 'mse':r'MSE'}
    for metrics_choice in metrics_list:
        fig = plt.figure(figsize=(7.5,6))
        y_label = label_list.get(metrics_choice,metrics_choice)
        plotting_col = [col for col in df.columns if metrics_choice in col]
        
        ini_col = [col for col in plotting_col if 'ini' in col]
        ini_x = int(ini_col[0].split('_')[-1])
        ini_y = df[ini_col]
        plt.scatter(ini_x,ini_y,label=r'Initial',c='black',marker='o',s=35)
        
        random_col = [col for col in plotting_col if 'random' in col]
        random_x = [int(col.split('_')[-1]) for col in random_col]
        random_y = df[random_col]
        plt.scatter(random_x,random_y,label=r'Random',marker='x',s=35)

        dt_col = [col for col in plotting_col if 'dt' in col]
        dt_x = [int(col.split('_')[-1]) for col in dt_col]
        dt_y = df[dt_col]
        plt.scatter(dt_x,dt_y,label=r'DT-guided',marker='v',s=35)

        gsx_col = [col for col in plotting_col if 'gsx' in col]
        gsx_x = [int(col.split('_')[-1]) for col in gsx_col]
        gsx_y = df[gsx_col]
        plt.scatter(gsx_x,gsx_y,label=r'GSx',marker='s',s=35)
        
        legend = plt.legend(fontsize=20,bbox_to_anchor=(1.07, 1.15),ncols=4,columnspacing=0.5,handletextpad=0.1)
        legend.get_frame().set_linewidth(1.5)
        legend.get_frame().set_edgecolor("black")
        plt.xlabel(r'Case Number',fontsize=25,fontweight='bold')
        plt.ylabel(f'{y_label}',fontsize=25,fontweight='bold')
        fig.savefig(os.path.join(PATH.fig_savepath,case_name,f'{model_choice}_{metrics_choice}_perf.png'))
        plt.show()

    print('Good job')


if __name__ == "__main__":
    main()