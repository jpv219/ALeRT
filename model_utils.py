##########################################################################
#### Model train and evaluate utilities
#### Author : Juan Pablo Valdes and Fuyue Liang
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import numpy as np
import pandas as pd
import pickle
import os
import shutil
import re
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from typing import Union
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import RepeatedKFold, KFold, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras.metrics import R2Score
from keras.models import Sequential, Model
from keras.layers import InputLayer, Dense, Input, Concatenate, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib
from paths import PathConfig
from kerastuner.tuners import Hyperband, RandomSearch, GridSearch, BayesianOptimization
from kerastuner import HyperParameters, Objective


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


############################ KFOLD CROSS VALIDATION ########################################

class KFoldEarlyStopping:

    def __init__(self, metric, patience = 5, delta = 0.001, verbose = True):
        """
        Args:
            delta (float): Porcentual minimum change in the monitored metric to be considered an improvement.
                            Default: 0.001
        """
        # stopper attributes
        self._counter = 0
        self._stop = False
        self.verbose = verbose
        self.patience = patience
        self.delta = delta

        # cv specific attributes
        self.score = metric

        # tracking best model overall and each kfold run performance: containing score, k_iteration, model
        self._best_score = None 
        self._best_k = 0
        self._best_model = None

    @property
    def stop(self):
        return self._stop
    
    @stop.setter
    def stop(self, value):
        if not isinstance(value, bool):
            raise ValueError("Stop value must be a boolean.")
        self._stop = value

    @property
    def counter(self):
        return self._counter
    
    @counter.setter
    def counter(self, value):
        if value < 0:
            raise ValueError("Counter value cannot be negative.")
        self._counter = value

    @property
    def best_score(self):
        return self._best_score
    
    @best_score.setter
    def best_score(self,value):
        self._best_score = value

    @property
    def best_model(self):
        return self._best_model
    
    @best_model.setter
    def best_model(self,value):
        self._best_model = value

    @property
    def best_k(self):
        return self._best_k
    
    @best_k.setter
    def best_k(self,value):
        self._best_k = value

    def print_verbose(self, message):
        if self.verbose:
            print(message)

    # getter for best pack
    def get_best_fold(self) -> list:
        return [self.best_score, self.best_k, self.best_model]

    # Call early stopping algorithm
    def __call__(self, current: list, mode = 'min'):

        # Update best results based on current fold results
        has_updated = self.__update_best_score(current, mode)

        _, current_k, _ = current
      
        # if no improvement exists, start counting towards the early stop
        if not has_updated:

            self.counter += 1

            self.print_verbose(f'Early stopping counter: {self.counter} out of {self.patience}')

        # Reset early stopping counter if an improvement exists: current = best
        else:
            self.counter = 0

        # If counter reaches patience, send stop signal
        if self.counter >= self.patience:

            self.stop = True

            self.print_verbose('-'*72)
            self.print_verbose(f'Stopping kfold sensitivity early stopping at fold {current_k}')

        
    # Evaluate kfold cv run performance vs. previous best kfold run to update best model so far
    def __update_best_score(self, current: list, mode = 'min') -> bool:
        
        has_updated = False

        if mode not in ['max', 'min']:
            raise ValueError('Unsupported update mode. Supported modes: "min", "max"')
        
        # Extract current values from kfold run instance
        current_results, current_k, current_model = current
        current_score = current_results[self.score]['mean']

        previous_best = self.best_score
        
        # Update best attributes if they are none or if an improvement is seen by a porcentual delta
        if self.best_score is None or \
           (mode == 'min' and current_score < self.best_score - self.delta*self.best_score) or \
           (mode == 'max' and current_score > self.best_score + self.delta*self.best_score):
            
            self.best_score = current_score
            self.best_k = current_k
            self.best_model = current_model
            has_updated = True

            self.print_verbose(f'Best scores updated at fold {current_k}: {self.score} now at {self.best_score} from {previous_best}')
        
        return has_updated
                
class KFoldCrossValidator(PathConfig):

    model_abbr_map = {'Decision_Tree':'dt', 
                    'XGBoost':'xgb', 
                    'Random_Forest': 'rf',
                    'Support_Vector_Machine': 'svm',
                    'K_Nearest_Neighbours': 'knn',
                    'MLP_Branched_Network': 'mlp_br',
                    'Multi_Layer_Perceptron': 'mlp'}
    
    def __init__(self, model, name: str, native: str, k_sens = True, verbose = True):

        super().__init__()
        
        self.model_name = name
        self.model_abbr = KFoldCrossValidator.model_abbr_map.get(name)
        
        self.model = model
        self.k_sens = k_sens
        self.verbose = verbose
        self.native = native

        self.chk_dir = os.path.join(self.model_savepath, self.model_name)

        self.bestmodel_path = ''
    
    def __call__(self, *args: Union[np.any, str], **kwargs: Union[np.any, str]) -> dict:
               
        # Training data sets
        X, y = args[0], args[1]

        # Optional kwargs depending on kfoldcv call
        cv_type = kwargs.get('cv_type')
        n_repeats = kwargs.get('n_repeats', 3)
        es_score = kwargs.get('earlystop_score', 'mse')

        # If cv_type is 'repeated', specify the number of repeats
        kfolders = {'kfold': KFold,
                'repeated': lambda n_splits: RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats),
                }

        # mode whether going for sknative or MLP kfold function
        if self.native == 'sk_native':
            cv_wrapper = self.sk_native_cv
        elif self.native == 'mlp':
            cv_wrapper = self.mlp_cv
        else:
            raise ValueError(f'Cross validator mode : {self.native} not supported. Options are sk_native or mlp')
         
        # Select Validator object based on input arguments
        if cv_type in kfolders.keys():
            cv_class = kfolders[cv_type]
        else:
            raise ValueError('Kfold cross validator type not specified or not supported, cv_type = str must be included in cross_validator call')
            
        # Initialize early stopping logic
        ## reminder: es_score is loss for mlp and mse for sk_native
        early_stopper = KFoldEarlyStopping(es_score, patience= 5, delta= 0.001, verbose = True)

        # run either a full k sensitivity cross validation or single kfold instance for a given k
        if self.k_sens:
            min_k = kwargs.get('min_k',3)
            max_k = kwargs.get('max_k', 50)

            kf_cv_summary = self.ksens_loop(X, y, cv_class, cv_wrapper, early_stopper, min_k, max_k)

        else:
            k = kwargs.get('k', 5)

            kf_cv_summary = self.kfold_cv(X,y,cv_class,cv_wrapper,k)

        return kf_cv_summary, self.bestmodel_path

    # one pass kfold crossvalidation
    def kfold_cv(self,X,y,cv_class,cv_wrapper, k) -> dict:
        
        # Checkpoint path
        self.clean_dir(self.chk_dir)
        
        #Cross validation set splitter
        cv = cv_class(n_splits = k)

        # calling native cv run instance
        kfold_results = cv_wrapper(X,y,cv,k)

        # Returning final mean metrics for best fold
        kf_cv_summary = {'k-fold': k}

        # extracting mean values from fold metrics
        for metric in kfold_results:
            kf_cv_summary[metric] = kfold_results[metric]['mean']
        
        # Save best model obtained
        self.save_model(self.model,k)

        # Save metrics log for all kfold runs carried out
        with open(os.path.join(self.chk_dir, f'{self.model_abbr}_kfold_cv_scores.txt'), 'w') as file :
            file.write(f'Results for cv run with k={k}:' + '\n')
            file.write('-'*72 + '\n')
            for metric in kfold_results.keys():
                file.write(f'{metric}: {kfold_results[metric]}' + '\n')
                file.write('-'*72 + '\n')
        
        return kf_cv_summary
    
    # k sensitivity loop kfold cross validation
    def ksens_loop(self, X, y, cv_class, cv_wrapper, early_stopper, min_k, max_k) -> dict:
        
        # Checkpoint path
        self.clean_dir(self.chk_dir)
        
        # number of kfolds to try as hyperparameter for the cross validation sensitivity
        folds = range(min_k,max_k+1)

        # List to store all metrics per kfold cross validation run
        cv_results = {f'kfold_{fold}':{} for fold in folds}
        
        for k in folds:

            self.print_verbose('-'*72)
            self.print_verbose(f'Starting Cross-Validation with {k} folds ...')

            #Cross validation set splitter
            cv = cv_class(n_splits = k)

            # call native cv run instance
            ## return a dictionary with min, max, mean for each metric
            kfold_results = cv_wrapper(X, y, cv, k)

            # Update overall results with k-fold cv instance results
            cv_results[f'kfold_{k}'].update(kfold_results)

            # Current kfold state
            current = [kfold_results, k, self.model]

            # Early stopping for kfold sensitivity
            early_stopper(current)

            if early_stopper.stop: 
                break

        # get best pack from early stopper
        _, best_fold_idx, best_model = early_stopper.get_best_fold()

        # Drop all future kfolds after algorithm has decided to early stop at a best kfold
        folds_to_drop = [f'kfold_{k}' for k in range(best_fold_idx+1, max_k+1)]
        for key in folds_to_drop:
            del cv_results[key]
        
        # save best model
        self.save_model(best_model,best_fold_idx)

        # Save metrics log for all kfold runs carried out
        with open(os.path.join(self.chk_dir, f'{self.model_abbr}_ksens_cv_scores.txt'), 'w') as file :
            for k, fold_run in enumerate(cv_results.keys()):
                file.write(f'Results for cv run with k={k+min_k}: {cv_results[fold_run]}' + '\n')
                file.write('-'*72 + '\n')
        
        # Returning final mean metrics for best fold
        kf_cv_summary = {'Best fold': best_fold_idx}

        for metric in cv_results[f'kfold_{best_fold_idx}']:
            kf_cv_summary[metric] = cv_results[f'kfold_{best_fold_idx}'][metric]['mean']
            
        return kf_cv_summary

    def sk_native_cv(self, X, y, cv, k) -> dict:
            
        scores_abs = {}
        scores = {}

        rename_keys = {'estimator':'estimator', 'fit_time': 'fit_time','score_time':'score_time',
                       'test_r2': 'r2', 'test_neg_mean_absolute_error': 'mae',
                       'test_neg_mean_squared_error': 'mse', 'test_explained_variance': 'variance',
                       'test_neg_root_mean_squared_error':'rmse'}
        
        # Extract detailed scores and performance per model
        score_metrics = ['explained_variance','r2','neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error']
        
        sk_scores = cross_validate(self.model, X, y, scoring=score_metrics,cv=cv, n_jobs=5, verbose=0, return_estimator=True) #number of folds X number of repeats

        # Store the overall metrics obtained for the k cross validation run tested
        for metric in sk_scores.keys():
            
            # avoid estimator objects from cross_validate
            if metric != 'estimator':
                # Take absolute value from sklearn natively negative metrics
                if 'neg' in metric:
                    scores_abs[metric] = np.abs(sk_scores[metric])
                else:
                    scores_abs[metric] = sk_scores[metric]
        
        # rename scores_abs metrics names for better readability in prints
        scores = {rename_keys.get(old_key): value for old_key, value in scores_abs.items()}

        # update self.model with estimator from best fold 
        best_fold_idx = np.argmin(scores['mse'])
        estimator = sk_scores['estimator'][best_fold_idx]
        self.model = estimator

        # calculate mean,max,min for all fold results metrics and return to caller
        kfold_results = self.store_fold_results(scores,k)
        
        return kfold_results

    def mlp_cv(self, X, y, cv, k) -> dict:

        # Containers to store results per k-fold cv run
        fold_results = {}

        # initialize val_loss threshold for ModelCheckpoint
        best_val_loss = float('inf')

        # Initialize checkpoint holder
        latest_checkpoint = None

        # keras callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        # Chkpt dir for each kfold instance run
        checkpoint_dir = os.path.join(self.model_savepath, self.model_name, f'{k}_fold_run')

        self.clean_dir(checkpoint_dir)

        # Loop over CV kfold repeats and extract average values per kfold run instance
        ## 'split' to generate indices to split data into traning and test set 
        for fold_idx, (train, test) in enumerate(cv.split(X,y)):
            
            self.print_verbose(f'Currently on fold {fold_idx} from k = {k} kfold run ...')
            # Creating callbacks for repeats within a kfold run
            checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold_idx}_best.keras')
            checkpoint = ModelCheckpoint(checkpoint_path, 
                                monitor='val_loss', save_best_only= True, 
                            verbose=0, mode='min', initial_value_threshold = best_val_loss)
            
            callbacks_list = [checkpoint,early_stopping]

            # Fit network with CV split train, val sets and call checkpoint callback
            ## pass validaton for monitoring validation loss and metrics at the end of each epoch
            ## the returned history holds a record of the loss values and metric value during training (loss and val_loss)
            ## callbacks are used to perform tasks: saving model checkpoint and early stopping (can be logging traning metrics as well)
            history = self.model.fit(X[train],y[train], 
                            validation_data=(X[test], y[test]), epochs = 50, batch_size = 1,
                            callbacks=callbacks_list,verbose=0)
            
            # Save repeat checkpoint if it has been created  - track last repeat checkpoint created
            if os.path.exists(checkpoint_path):
                latest_checkpoint = checkpoint_path
            
            # Load weights from last repeat checkpoint saved
            self.model.load_weights(latest_checkpoint)

            # Evaluate model fit on validation set according to Kfold split
            ## the retured cores are specified when compiling the model
            scores = self.model.evaluate(X[test], y[test], verbose=0)

            # Update best val_loss obtained from all repeats in the present kfold run
            if scores[0]< best_val_loss:
                best_val_loss = scores[0]
            
            # Save network metrics previously compiled in build_net() per repeat executed
            for i, metric in enumerate(history.history.keys()):

                # Only loop through the history metrics included in the scores
                if i < len(scores):
                    # initialize or append metric values per repeat 
                    if metric in fold_results:
                        fold_results[metric].append(scores[i])
                    else:
                        fold_results[metric] = [scores[i]]
                else:
                    break

        self.print_verbose('-'*72)

        # calculate mean,max,min for all fold results metrics and return to caller
        kfold_results = self.store_fold_results(fold_results,k)

        return kfold_results
    
    def store_fold_results(self,scores,k) -> dict:
        
        # Container to store results per k-fold cv run
        kfold_results = {}
        
        # Store the overall metrics obtained for the k cross validation run tested
        for metric in scores.keys():
            kfold_results[metric] = {'mean': np.mean(scores[metric]),
                                        'min': np.min(scores[metric]),
                                        'max': np.max(scores[metric])}

        self.print_verbose('-'*72)
        for metric in kfold_results.keys():
            self.print_verbose(f'Mean scores with {k} folds: {metric} = {kfold_results[metric]["mean"]};' )
            
        self.print_verbose('-'*72)

        return kfold_results
    
    def save_model(self,best_model,k_idx):

        # Save best model obtained
        if self.native == 'sk_native':
            chk_path = os.path.join(self.chk_dir, f'{self.model_abbr}_best_{k_idx}_fold.pkl')
            joblib.dump(best_model,chk_path)

        elif self.native == 'mlp':
            chk_path = os.path.join(self.chk_dir, f'{self.model_abbr}_best_{k_idx}_fold.keras')
            best_model.save(chk_path)
        
        # update best model path
        self.bestmodel_path = chk_path

    def print_verbose(self, message):
        if self.verbose:
            print(message)

    @staticmethod
    def clean_dir(dir):
        
        # Create kfold run checkpoint folder
        if not os.path.exists(dir):
            os.makedirs(dir)

        # clean if previous files exist
        else:
            for filename in os.listdir(dir):
                file_path = os.path.join(dir, filename)

                # Check if the current file is a directory
                if os.path.isdir(file_path):
                    # Check if the directory is 'hyperparam_tune'
                    if filename == 'hyperparam_tune':
                        continue  # Skip this directory
                    else:
                        shutil.rmtree(file_path)  # Remove directory and its contents
                else:
                    os.remove(file_path)

############################ HYPERPARAMETER TUNING ########################################

class HyperParamTuning(PathConfig):

    ## SEARCH SPACES ##
    regressor_hp_search_space = {'dt': {'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                'max_depth': [2, 4, 6, 8, 10],
                'min_samples_split': [2, 4, 6, 8, 10, 12, 14],
                'min_samples_leaf': [1, 2, 3, 4, 6],
                'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3],
                'max_leaf_nodes': [None, 2, 5],
                'splitter' : ['best','random']}, 
        'xgb': {'max_depth': [1,3,6,9,12], 'n_estimators': [100,150,200],
                'learning_rate': [0.01,0.05,0.1,0.3], 'min_child_weight': [1,3,5,7,9],
                'subsample': [0.5,0.7, 1], 'colsample_bytree': [0.5, 0.7, 1.0],
                'gamma': [0, 0.01, 0.05], 'lambda' : [0.001, 0.01, 0.05],
                'alpha': [0.05, 0.1, 0.5]}, 
        'rf': {'n_estimators': [100,200,400,600,800,1000],
               'max_depth': [None, 10, 20, 40, 60],
               'min_samples_split': [2,5,10,20],
               'max_features': [1,'sqrt','log2'],
               'min_samples_leaf': [1,2,4,8],
               'bootstrap': [True,False]},
        'svm': {'estimator__C': [0.01,0.1,1,10,50], 
                'estimator__epsilon': [0.001,0.01,0.1],
                'estimator__kernel': ['linear','poly','rbf','sigmoid'],
                'estimator__gamma': ['scale','auto',0.001,0.01,0.1],
                'estimator__degree': [2,3,4],
                'estimator__coef0': [0,0.1,1]},
        'knn': {'n_neighbors': [int(x) for x in range(1,12)],
                'weights': ['uniform','distance'],
                'p': [1,2,3],
                'algorithm': ['auto','ball_tree','kd_tree','brute'],
                'leaf_size': [10,30,50],
                'metric': ['euclidean', 'minkowski','chebyshev']},
        'mlp_br': {'n_nodes_1' : (64,512,32),
                'n_nodes_2': (32,192,32),
                'n_nodes_br': (32,256,32),
                'act_fn': ['relu','sigmoid', 'tanh'],
                'lr': [1e-2, 1e-3, 1e-4]},
        'mlp': {'n_dense_layers' : (1,5,1), # when using tuples, the values specified are initial,final,step
                'n_shallow_layers': (1,5,1),
                'n_nodes_dense': (64,512,32),
                'n_nodes_shallow': (32,128,32),
                'act_fn': ['relu', 'sigmoid', 'tanh'],
                'lr': [1e-2, 1e-3, 1e-4]}
    }

    key_regressor_params = {'dt': ['max_depth','min_samples_split'],
                            'xgb': ['max_depth', 'min_child_weight', 'learning_rate'],
                            'rf': ['n_estimators','max_depth','min_samples_split'],
                            'svm': ['estimator__kernel','estimator__C','estimator__gamma'],
                            'knn': ['n_neighbors', 'weights']}

    model_abbr_map = {'Decision_Tree':'dt', 
                    'XGBoost':'xgb', 
                    'Random_Forest': 'rf',
                    'Support_Vector_Machine': 'svm',
                    'K_Nearest_Neighbours': 'knn',
                    'MLP_Branched_Network': 'mlp_br',
                    'Multi_Layer_Perceptron': 'mlp'}
    
    rename_keys = {
            'r2': 'r2',
            'mae': 'neg_mean_absolute_error',
            'mse': 'neg_mean_squared_error',
            'variance': 'explained_variance',
            'rmse': 'neg_root_mean_squared_error',
            'loss': 'loss'
        }
    
    
    def __init__(self, model, name, native, verbose = False):

        super().__init__()
        
        self.model_name = name
        self.model_abbr = self.get_value(dict_name= 'model_abbr_map',key= name)
        self.native = native
        self.verbose = verbose

        self.model = model

        if self.model_abbr is None:
            raise NotImplementedError('Model not supported for Hyperparameter tuning')
    
    def __call__(self, *args: Union[np.any, str], **kwargs: str) :
        
        # Training data sets
        X, y = args[0], args[1]

        # Optional kwargs depending on tuning cv call and native model
        if self.native == 'sk_native':
            tuning_type = kwargs.get('sk_tuning_type')
        else:
            tuning_type = kwargs.get('mlp_tuning_type')
        input_score = kwargs.get('fit_score')
        n_iter  = kwargs.get('n_iter', None)
        max_trials = kwargs.get('max_trials', None)

        # get sklearn appropaite identifier for fit score
        fit_score = self.get_value(dict_name='rename_keys', key= input_score)

        # get hyperparameter searcher
        param_grid = self.get_value(dict_name='regressor_hp_search_space',key = self.model_abbr)

        # create/clean saving tune directory
        tune_save_dir = os.path.join(self.model_savepath,self.model_name,'hyperparam_tune')
        self.clean_dir(tune_save_dir)

        self.print_verbose('-'*72)
        self.print_verbose(f'Running hyperparameter tuning for {self.model_name} with tuner: {tuning_type}')
        self.print_verbose('-'*72)

        # mode whether going for sknative or MLP hyperparameter tune function
        if self.native == 'sk_native':
            tuned_model = self.sk_native_tuner(X, y, tuning_type, param_grid, fit_score, n_iter)

            # Get best parameters and best estimator
            best_params = tuned_model.best_params_
            best_estimator = tuned_model.best_estimator_
            best_score = tuned_model.best_score_
            results_df = pd.DataFrame(tuned_model.cv_results_)

            # extract score column to rank best trials executed during search
            rank_column = next((col for col in results_df.columns if col == 'rank_test_' + fit_score), 
                    next((col for col in results_df.columns if col.startswith('rank_test_')), None))
            sorted_results = results_df.sort_values(by=rank_column, ascending= True)

            # save best performing model and parameter detail to a txt file
            with open(os.path.join(tune_save_dir,f'{self.model_abbr}_tune_summary.txt'), 'w') as file:
                file.write(f'Results summary for top 5 cases during hyperparameter tuning search with {tuning_type} tuner' + '\n')
                
                for column in sorted_results.columns:
                    #write only top 5 cases from sorted dataframe
                    for i in range(len(sorted_results[:5])):
                        file.write(f'{column} for case # {i}: {sorted_results[column].iloc[i]}' + '\n')
                    file.write('-'*72 + '\n')

                file.write('Best Parameters overall:' + '\n')
                file.write('-'*72 + '\n')
                file.write(f'{best_params}')

            self.print_verbose('-'*72)
            self.print_verbose(f'Best Parameters: {best_params}')
            self.print_verbose(f'Best Score at Tuning: {-best_score}')
            self.print_verbose('-'*72)
        
        elif self.native == 'mlp':

            input_size = kwargs.get('input_size')
            output_size = kwargs.get('output_size')
            n_features = kwargs.get('n_features')

            best_estimator, tuner = self.mlp_hp_tuner(X, y, tuning_type, param_grid, fit_score, max_trials,
                                                      input_size, output_size,n_features)

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            # save best performing model and parameter detail to a txt file
            with open(os.path.join(tune_save_dir,f'{self.model_abbr}_tune_summary.txt'), 'w') as file:
                file.write(f'Results summary for top 5 cases during hyperparameter tuning search with {tuning_type} tuner' + '\n')
                file.write('-'*72 + '\n')

                for trial in tuner.oracle.get_best_trials(5):
                    file.write(f'Trial ID: {trial.trial_id}' + '\n')
                    file.write(f'Hyperparameters: {trial.hyperparameters.values}' + '\n')
                    file.write(f'Score: {trial.score}' + '\n')
                    file.write('-'*72 + '\n')

                file.write('Best Parameters overall:' + '\n')
                file.write('-'*72 + '\n')
                for item in param_grid.keys():
                    file.write(f'best {item}: {best_hps.get(item)}' + '\n')

            self.print_verbose('-'*72)
            tuner.results_summary()
            self.print_verbose('-'*72)
        
        return best_estimator
    
    def print_verbose(self, message):
        if self.verbose:
            print(message)

    @staticmethod
    def clean_dir(dir):
        
        # Create kfold run checkpoint folder
        if not os.path.exists(dir):
            os.makedirs(dir)

        # clean if previous files exist
        else:
            for filename in os.listdir(dir):
                file_path = os.path.join(dir,filename)

                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
                else:
                    os.remove(file_path)

    def sk_native_tuner(self, X: np.array, y: np.array, tuning_type: str, param_grid: dict, fit_score: str, n_iter: Union[int, None]):
        
        # Select type of hyperparameter tuning process to execute
        tuners = {'std': GridSearchCV,
                'random': RandomizedSearchCV,
                'halving': HalvingGridSearchCV,
                'halve_random': HalvingRandomSearchCV
                }

        hp_tuner = tuners.get(tuning_type, None)

        if hp_tuner is None:
            raise NotImplementedError(f'hyperparam searching method specified: {tuning_type} is not supported')
        
        # Extract detailed scores and performance per model
        score_metrics = ['explained_variance','r2','neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error']

        # First tuning run on most influential parameters
        key_params = self.get_value(dict_name = 'key_regressor_params',key = self.model_abbr)
        first_param_sweep = {key: param_grid[key] for key in key_params}

        self.print_verbose('-'*72)
        self.print_verbose(f'Starting first hyperparameter sweep for {self.model_name} with parameters: {key_params}')
        self.print_verbose('-'*72)
        
        first_search = self.sk_run_tune(first_param_sweep, fit_score, score_metrics, tuner = tuners.get('std'), tuning_type='std', n_iter = None)

        # Fit model with hyperparam tuning search
        first_tune = first_search.fit(X,y)

        # Extract best parameters from first tune sweep
        best_key_params = {param : [first_tune.best_params_[param]] for param in first_param_sweep.keys()}

        self.print_verbose('-'*72)
        self.print_verbose(f'Continuing final sweep for {self.model_name} with remaining parameters')
        self.print_verbose('-'*72)

        # Re-build sample space with key parameters as constants from initial sweep
        full_param_sweep = {key: best_key_params[key] if key in best_key_params else value for key, value in param_grid.items()}

        # Full tuning sweep with constant best parameters from first sweep
        search = self.sk_run_tune(full_param_sweep, fit_score, score_metrics, hp_tuner, tuning_type, n_iter)

        # Fit model with hyperparam tuning search
        final_tune = search.fit(X,y)
        
        return final_tune
    
    def sk_run_tune(self, params: dict, fit_score: str, score_metrics: list, tuner, tuning_type: str, n_iter: Union[int, None]):

        if 'halv' in tuning_type:
            search = tuner(self.model, params, scoring = fit_score, n_jobs = -1, cv = 3, verbose = 2)

        elif tuning_type == 'random':
            search = tuner(self.model, params, scoring = fit_score, n_iter = n_iter, n_jobs = -1, cv = 3, verbose = 2)

        else:
            search = tuner(self.model, params, scoring = score_metrics, n_jobs = -1, refit = fit_score, cv = 3, verbose = 2, error_score = 'raise')

        return search
    
    def mlp_hp_tuner(self, X: np.array, y: np.array, tuning_type: str, param_grid: dict, fit_score: str, max_trials: Union[int, None],
                     input_size: int, output_size: int, n_features: int):

        #save directory for tuning trials to be stored
        save_dir = os.path.join(self.model_savepath,self.model_name)

        #construct hyperparameter sample space to be explored by Keras tuner
        hp = HyperParameters()

        for param, values in param_grid.items():

            # specific values in a list
            if isinstance(values, list):
                hp.Choice(param,values)

            # values defined as a step-wise list
            elif isinstance(values, tuple):
                hp.Int(param,values[0],values[1],values[2])

        # build network function set as partial to hand in data shape inputs
        build_net_partial = partial(self.build_net, input_size = input_size, 
                                    output_size = output_size, n_features = n_features)

        # Select tuner from Keras tuner library

        tuner_args = {
            'hypermodel': build_net_partial,
            'objective': Objective(fit_score, 'min'),
            'max_trials': max_trials,
            'hyperparameters': hp,
            'directory': save_dir,
            'project_name': 'hyperparam_tune'
        }

        if tuning_type == 'hyperband':
            
            # Include specific settings for hyperband tuner and remove max_trials
            tuner_args.update({
                'max_epochs' : 100,
                'factor': 3,
                'hyperband_iterations': 3
            })
            tuner_args.pop('max_trials')
            
            tuner = Hyperband(**tuner_args)

        elif tuning_type == 'random':

            tuner = RandomSearch(**tuner_args)

        elif tuning_type == 'grid_search':
            
            tuner = GridSearch(**tuner_args)

        elif tuning_type == 'bayesian':

            tuner_args['beta'] = 5
            
            tuner = BayesianOptimization(**tuner_args)

        stop_early = EarlyStopping(monitor=fit_score, patience=10)

        # Perform the hyperparameter search
        tuner.search(X, y, 
                     validation_split=0.3, 
                     epochs=50,
                     shuffle=True,
                     callbacks=[stop_early]
                     )

        # get best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the best model
        model = tuner.hypermodel.build(best_hps)

        # Train the best model and find the best performing epoch
        history = model.fit(X, y, epochs=100, validation_split = 0.3, batch_size = 1)

        val_loss_per_epoch = history.history['val_loss']
        best_epoch = np.argmax(val_loss_per_epoch) + 1

        self.print_verbose(f'Re-training the model with the optimal epochs ({best_epoch}) and hps found')
        
        #re-train the model with the optimal epoch found
        best_model = tuner.hypermodel.build(best_hps)

        best_model.fit(X, y, epochs=best_epoch, validation_split = 0.3, batch_size = 1, callbacks = [stop_early])

        return best_model, tuner

    def build_net(self, hp, input_size, output_size, n_features):
        """
        Build network model for hyperparameters tuning

        hp: HyperParameters class instance
        """
        lr = hp.get('lr')
        
        if self.model_abbr == 'mlp':
            net = self.mlp(hp,input_size,output_size)
        else:
            net = self.mlp_branched(hp,input_size,output_size, n_features)

        # Network training utilities
        optimizer = Adam(learning_rate=lr)

        net.compile(optimizer= optimizer, loss = 'mean_squared_error', metrics = ['mae', 'mse', R2Score()])

        return net

    @staticmethod
    def mlp(hp, input_size, output_size):
        
        net = Sequential()
        
        #Hyperparams
        n_dense_layers = hp.get('n_dense_layers')
        n_shallow_layers = hp.get('n_shallow_layers')
        n_nodes_dense = hp.get('n_nodes_dense')
        n_nodes_shallow = hp.get('n_nodes_shallow')
        act_fn = hp.get('act_fn')

        # Input layer
        net.add(InputLayer(shape=(input_size,)))

        # Dense layers, with more nodes per layer
        for _ in range(n_dense_layers):
            net.add(Dense(n_nodes_dense,activation=act_fn))

        # Shallow layers, with less nodes per layer
        for _ in range(n_shallow_layers):
            net.add(Dense(n_nodes_shallow,activation=act_fn))

        # Output layer
        net.add(Dense(output_size,activation='linear'))

        return net

    @staticmethod
    def mlp_branched(hp, input_size, output_size, n_features):

        #Hyperparams
        n_nodes_1 = hp.get('n_nodes_1')
        n_nodes_2 = hp.get('n_nodes_2')
        n_nodes_br = hp.get('n_nodes_br')
        act_fn = hp.get('act_fn')
        
        inputs = Input(shape=(input_size,))

        # hidden layers for processing inputs
        hidden1 = Dense(n_nodes_1, activation=act_fn)(inputs)
        hidden2 = Dense(n_nodes_2, activation= act_fn)(hidden1)

        #construct branches for each feature and connect to output
        outputs = []

        for _ in range(n_features):

            branch_hidden = Dense(n_nodes_br, activation= act_fn) (hidden2)
            branch_out = Dense(100, activation= 'linear')(branch_hidden)
            outputs.append(branch_out)

        concatenated = Concatenate()(outputs)

        reshaped_out = Reshape((output_size,))(concatenated)

        net = Model(inputs = inputs, outputs = reshaped_out)

        return net
    
    @classmethod
    def get_value(cls, dict_name : str, key: str):

        # check if dictionary exists
        if not hasattr(cls, dict_name):
            raise ValueError(f'Dictionary {dict_name} does not exist in {cls.__name__}')
        
        # Retrive class dictionary
        dictionary = getattr(cls, dict_name)

        if key not in dictionary:
            raise KeyError(f'Key {key} specified does not exist in dictionary {dict_name}')
        
        return dictionary.get(key)


########################### MODEL EVALUATION ##############################################
                    
class ModelEvaluator(PathConfig):

    def __init__(self, model, modelname:str, data_packs: list,case: str, pca:bool, datasample: str):
        super().__init__()

        self.model = model
        self.modelname = modelname
        self._case = case
        self.pca = pca
        self.datasample = datasample

        # Reading data packs for model fit and eval
        self.X_train_df, self.y_train_df, self.X_test_df, self.y_test_df = data_packs[:4]
        
        # Converting to numpy arrays for plotting and further processing
        self.X_train = self.X_train_df.to_numpy()
        self.y_train = self.y_train_df.to_numpy()
        self.X_test = self.X_test_df.to_numpy()
        self.y_test = self.y_test_df.to_numpy()


    def inverse_pca(self, y_pred, y_target_df):
        # allocate the columns on the prediction array
        y_pred_df = pd.DataFrame(y_pred, columns=self.y_train_df.columns)
        # extract all the reduced features, e.g., Q, E_max
        pca_features = set([col.split('_pc')[0] for col in y_pred_df.columns])
        
        # load the saved pca components for each feature
        for feature in pca_features:
            with open(os.path.join(self.pca_savepath, self._case, self.datasample, f'pca_model_{feature}.pkl'), 'rb') as f:
                pca_compnts_per_feat = pickle.load(f)

            # extract all the columns related one feature
            y_pred_per_feat = y_pred_df.filter(regex=f'{feature}_pc')
            # inverse tranform the reduced feature back to normal space
            y_invpred_per_feat = pca_compnts_per_feat.inverse_transform(y_pred_per_feat)
            # allocate the columns of normal space and drop the column of reduced space
            y_invpred_per_feat.columns = [f'{feature}'+'_{}'.format(i) for i in range(y_invpred_per_feat.shape[1])]
            y_pred_df = pd.concat([y_pred_df, y_invpred_per_feat],axis=1).drop(y_pred_per_feat.columns,axis=1)

        # align the inverse dataframe with the order of target data
        y_pred_df_align = y_pred_df[y_target_df.columns]
        y_pred_inv = y_pred_df_align.to_numpy()

        return y_pred_inv
    
    def predict(self,X_df,y_target_df):

        X = X_df.to_numpy()
        y_pred = self.model.predict(X)

        if self.pca and y_pred.shape != y_target_df.shape:
            y_pred_inv = self.inverse_pca(y_pred, y_target_df)
            return y_pred_inv
        else:
            return y_pred
 

    def plot_overall_dispersion(self):

        y_pred_test = self.predict(self.X_test_df, self.y_test_df)
        y_pred_train = self.predict(self.X_train_df,self.y_train_df)

        y_list = [self.y_train, self.y_test]
        y_pred_list = [y_pred_train,y_pred_test]
        y_label_list = ['Training', 'Testing']
        color_list = ['blue','orange']
        
        fig,ax = plt.subplots(figsize=(8,6))

        for i in range(len(y_label_list)):
            
            x_min = min(np.min(y_pred_list[i]), np.min(y_list[i]))
            x_max = max(np.max(y_pred_list[i]), np.max(y_list[i]))
            x = np.linspace(x_min,x_max,100)

            ax.scatter(y_list[i][::25],y_pred_list[i][::25],marker='+',c=color_list[i],label=f'{y_label_list[i]} Data')#edgecolor='k', c= y_list[i], cmap=COLOR_MAP)
            
        # y=x
        ax.plot(x, x,label = r'y=x', color = 'k', linewidth = 2, linestyle='--')
        # deviation band
        dispersion = 0.2*np.abs(x)
        plt.fill_between(x,x-dispersion,x+dispersion,color='gray',alpha=0.2,label=r'$20\%$ Dispersion')

        ax.set_xlabel(r'True Data',fontweight='bold',fontsize=30)
        ax.set_ylabel(r'Predicted Data',fontweight='bold',fontsize=30)
        ax.tick_params(axis='both',labelsize=20)
    
        ax.legend(fontsize=20)
        # ax.set_title(f'{y_label_list[i]} Data',fontweight='bold',fontsize=30)
        fig.tight_layout()
        fig.savefig(os.path.join(self.fig_savepath, self._case, self.datasample, f'{self.modelname}_Pred.png'),dpi=200)
        plt.show()
    
    def plot_features_r2(self):

        y_pred = self.predict(self.X_test_df, self.y_test_df)

        # Calculate the r2 for each feature
        # empty dictionary to store the r2 for all features
        r2_values = {}
        # allocate the columns on the prediction array
        y_pred_df = pd.DataFrame(y_pred, columns=self.y_test_df.columns)
        # extract all the reduced features, e.g., Q, E_max
        pca_features = set(['_'.join(col.split('_')[:-1]) for col in y_pred_df.columns])
        # calculate the r2 value for each features
        fig1,ax = plt.subplots(figsize=(8,6))
        colors = sns.color_palette('muted',len(pca_features))
        feat_label = {'E_diss': r'$E_{diss}$',
                      'E_max': r'$e_{\lambda, max}$',
                      'Gamma': r'$\dot{\gamma}$',
                      'Velocity': r'$U_{av}$',
                      'Q':r'$Q$'}
        feat_colors = {key: colors[i] for i, key in enumerate(feat_label)}

        legend_handles = []

        legend_order = ['E_diss', 'E_max', 'Gamma', 'Velocity', 'Q']

        for idx, feat in enumerate(pca_features):
            pattern = re.compile(f'^{feat}_\d+$')
            # extract all the columns related one feature
            column_per_feat = [col for col in y_pred_df.columns if pattern.match(col)]
            # slicing associated columns for y_pred_df and y_test_df
            y_pred_slice = y_pred_df[column_per_feat].to_numpy()
            y_test_slice = self.y_test_df[column_per_feat].to_numpy()
            r2 = r2_score(y_test_slice,y_pred_slice)
            r2_values[feat] = r2

            # plot a dipsersion plot for each feature
            scatter = ax.scatter(y_test_slice[::25],y_pred_slice[::25],edgecolor='k',color= feat_colors.get(feat),label=f'{feat_label.get(feat,feat)}')
            legend_handles.append((scatter, feat_label.get(feat, feat)))
        
        # Plot y=x line
        x = np.linspace(-1, 1, 100)
        ax.plot(x, x, label=r'y=x', color='k', linewidth=2, linestyle='--')

        # Plot deviation band
        dispersion = 0.2 * np.abs(x)
        plt.fill_between(x, x - dispersion, x + dispersion, color='gray', alpha=0.2, label=r'$20\%$ Dispersion')

        ax.set_xlabel(r'True Data',fontweight='bold',fontsize=25)
        ax.set_ylabel(r'Predicted Data',fontweight='bold',fontsize=25)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)

        # Create custom legend in desired order
        ordered_handles = []
        ordered_labels = []

        for feat in legend_order:
            label = feat_label[feat]
            handle = next(h for h, l in legend_handles if l == label)
            ordered_handles.append(handle)
            ordered_labels.append(label)
        
        ax.legend(ordered_handles, ordered_labels, fontsize=18, handletextpad=0.2)

        fig1.savefig(os.path.join(self.fig_savepath, self._case, self.datasample,f'{self.modelname}_Pred_test.png'),dpi=200)
        plt.show()
        
        # # Sort the keys based on the r2 values in descending order, Overall always at the last
        # sorted_keys = sorted(r2_values.keys(),key=lambda x: r2_values[x], reverse=True)
        # r2_values['Overall'] = r2_score(self.y_test, y_pred)
        # sorted_feat = sorted_keys + ['Overall']
        # sorted_values = [r2_values[key] for key in sorted_feat]

        # fig2 = plt.figure(figsize=(8,6))
        # plt.bar(sorted_feat,sorted_values, edgecolor='black')
        # # Add the values at the top of bars
        # for i, value in enumerate(sorted_values):
        #     plt.text(i, value, f'{value:.3f}', ha='center',va='bottom',fontweight='bold')
        # plt.ylabel(r'$R^2$',fontsize=30)
        # plt.xticks(rotation=45)
        # plt.tick_params(axis='x',labelsize=20)
        # plt.tick_params(axis='y',labelsize=20)
        # fig2.tight_layout()
        # fig2.savefig(os.path.join(self.fig_savepath,self._case, self.datasample,f'{self.modelname}_R2.png'),dpi=200)
        # plt.show()

    def display_metrics(self):

        y_pred = self.predict(self.X_test_df, self.y_test_df)

        r2 = r2_score(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)

        print('-'*72)
        print(f'{self.modelname} Performance with {self.datasample} sampling')
        print('R2 Score: ', r2)
        print('Mean Squared Error: ', mse)
        print('Mean Absolute Error: ', mae)

