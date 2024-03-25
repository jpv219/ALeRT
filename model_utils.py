##########################################################################
#### Model train and evaluate utilities
#### Author : Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import numpy as np
import pandas as pd
import configparser
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Union
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import RepeatedKFold, KFold, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib


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

############################# PATH UTILITIES ##############################################
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
    
    @property
    def model_savepath(self):
        return self._config['Path']['models']

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
                    'Support_Vector_Machine': 'rf',
                    'K_Nearest_Neighbours': 'knn',
                    'MLP_Wrapped_Regressor': 'mlp_reg',
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
        for fold_idx, (train, test) in enumerate(cv.split(X,y)):
            
            self.print_verbose(f'Currently on fold {fold_idx} from k = {k} kfold run ...')
            # Creating callbacks for repeats within a kfold run
            checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold_idx}_best.keras')
            checkpoint = ModelCheckpoint(checkpoint_path, 
                                monitor='val_loss', save_best_only= True, 
                            verbose=0, mode='min', initial_value_threshold = best_val_loss)
            
            callbacks_list = [checkpoint,early_stopping]

            # Fit network with CV split train, val sets and call checkpoint callback
            history = self.model.fit(X[train],y[train], 
                            validation_data=(X[test], y[test]), epochs = 50, batch_size = 1,
                            callbacks=callbacks_list,verbose=0)
            
            # Save repeat checkpoint if it has been created  - track last repeat checkpoint created
            if os.path.exists(checkpoint_path):
                latest_checkpoint = checkpoint_path
            
            # Load weights from last repeat checkpoint saved
            self.model.load_weights(latest_checkpoint)

            # Evaluate model fit on validation set according to Kfold split
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
                file_path = os.path.join(dir,filename)

                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
                else:
                    os.remove(file_path)

############################ HYPERPARAMETER TUNING ########################################

class HyperParamTuning(PathConfig):

    ## SEARCH SPACES ##
    regressor_hp_search_space = {'dt': {'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                'max_depth': [2, 4, 6, 8],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'min_impurity_decrease': [0.0, 0.1, 0.2]}, 
        'xgb': {'max_depth': [3,5,7,9], 'n_estimators': [100,200,300,400,500],
                'learning_rate': [0.001,0.01,0.1,0.3], 'min_child_weight': [1,3,5,7],
                'subsample': [0.5,0.7,1], 'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3, 0.4], 'lambda' : [0.01, 0.1, 1.0],
                'alpha': [0.01, 0.1, 1.0]}, 
        'rf': {'n_estimators': 100},
        'svm': {'C': 1, 'epsilon': 0.1},
        'knn': {'n_neighbours': 10},
        'mlp_reg': {'n_dense' : 2,
                'n_shallow': 2,
                'n_nodes_d': 128,
                'n_nodes_s': 64,
                'n_epochs' : 100,
                'batch_size' : 1,
                'act_fn': 'relu'},
        'mlp': {'n_dense' : 2,
                'n_shallow': 2,
                'n_nodes_d': 128,
                'n_nodes_s': 64,
                'n_epochs' : 100,
                'batch_size' : 1,
                'act_fn': 'relu'}

    }

    model_abbr_map = {'Decision_Tree':'dt', 
                    'XGBoost':'xgb', 
                    'Random_Forest': 'rf',
                    'Support_Vector_Machine': 'rf',
                    'K_Nearest_Neighbours': 'knn',
                    'MLP_Wrapped_Regressor': 'mlp_reg',
                    'Multi_Layer_Perceptron': 'mlp'}
    
    rename_keys = {
            'r2': 'r2',
            'mae': 'neg_mean_absolute_error',
            'mse': 'neg_mean_squared_error',
            'variance': 'explained_variance',
            'rmse': 'neg_root_mean_squared_error'
        }
    
    
    def __init__(self, model, name, native):

        super().__init__()
        
        self.model_name = name
        self.model_abbr = HyperParamTuning.model_abbr_map.get(name, None)
        self.native = native

        self.model = model

        if self.model_abbr is None:
            raise NotImplementedError('Model not supported for Hyperparameter tuning')
    
    def __call__(self, *args: Union[np.any, str], **kwargs: str) :
        
        # Training data sets
        X, y = args[0], args[1]

        # Optional kwargs depending on tuning cv call
        tuning_type = kwargs.get('tuning_type')
        input_score = kwargs.get('fit_score', 'mse')
        n_iter  = kwargs.get('n_iter',None)

        # get sklearn appropaite identifier for fit score
        fit_score = HyperParamTuning.rename_keys.get(input_score)
        # get hyperparameter searcher
        param_grid = HyperParamTuning.regressor_hp_search_space.get(self.model_abbr)

        # create/clean saving tune directory
        tune_save_dir = os.path.join(self.model_savepath,self.model_name,'hyperparam_tune')
        self.clean_dir(tune_save_dir)

        # mode whether going for sknative or MLP hyperparameter tune function
        if self.native == 'sk_native':
            search = self.sk_native_tuner(tuning_type,param_grid,fit_score, n_iter)

        elif self.native == 'mlp':
            search = self.mlp_hp_tuner(param_grid)

        # Fit model with hyperparam tuning search
        tuned_model = search.fit(X,y)

        # Get best parameters and best estimator
        best_params = tuned_model.best_params_
        best_estimator = tuned_model.best_estimator_
        best_score = tuned_model.best_score_
        results_df = pd.DataFrame(tuned_model.cv_results_)

        # extract score column to rank best trials executed during search
        rank_column = [col for col in results_df.columns if col == 'rank_test_' + fit_score or col.startswith('rank_test_')]
        sorted_results = results_df.sort_values(by=rank_column)

        # save best performing model and parameter detail to a txt file
        with open(os.path.join(tune_save_dir,f'{self.model_abbr}_tune_summary.txt'), 'w') as file:
            file.write(f'Results summary for top 5 cases during hyperparameter tuning search with {tuning_type} tuner' + '\n')
            
            for column in sorted_results.columns:
                #write only top 5 cases from sorted dataframe
                for i in range(len(sorted_results[:5])):
                    file.write(f'{column}: {sorted_results[column][i]}' + '\n')
                file.write('-'*72 + '\n')

        print("Best Parameters:", best_params)
        print("Best Score at Tuning:", -best_score)

        return(best_estimator)

    def sk_native_tuner(self, tuning_type: str, param_grid, fit_score: str, n_iter = 1000):
        
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

        if 'halv' in tuning_type:
            search = hp_tuner(self.model, param_grid, scoring = fit_score, n_jobs = -1, cv = 3, verbose = 2)
        elif tuning_type == 'random':
            search = hp_tuner(self.model, param_grid, scoring = fit_score, n_iter = n_iter, n_jobs = -1, cv = 3, verbose = 2)
        else:
            search = hp_tuner(self.model, param_grid, scoring = score_metrics, n_jobs = -1, refit = fit_score, cv = 3, verbose = 2)

        return search
    
    def mlp_hp_tuner(self):
        pass

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


########################### MODEL EVALUATION ##############################################
                    
class ModelEvaluator(PathConfig):

    def __init__(self, model, x_test_df: pd.DataFrame, y_test_df: pd.DataFrame):
        super().__init__()

        self.model = model
        self.x_test_df = x_test_df
        self.y_test_df = y_test_df

        self.x_test = x_test_df.to_numpy()
        self.y_test = y_test_df.to_numpy()

        self.y_pred = None

    def predict(self):
        self.y_pred = self.model.predict(self.x_test)

    def plot_dispersion(self):

        if self.y_pred is None:
            self.predict()

        x = np.linspace(-1,1,100)
        y = x
        pos_dev = -1 +1.2*(x+1)
        neg_dev = -1 +0.8*(x+1)

        plt.figure(figsize=(8,6))
        plt.plot(x,y,label = 'x=y', color = 'k', linewidth = 2.5)
        plt.plot(x,pos_dev, label = '+20%', color = 'r', linewidth = 1.5, linestyle = '--')
        plt.plot(x,neg_dev, label = '-20%', color = 'r', linewidth = 1.5, linestyle = '--')
        plt.scatter(self.y_test, self.y_pred, edgecolor='k', c= self.y_test, cmap=COLOR_MAP)
        plt.xlabel('True Data')
        plt.ylabel('Predicted Data')
        plt.title('True vs. Pred dispersion plot')
        plt.legend()
        plt.show()
    
    def plot_r2_hist(self, num_bins = 10):

        if self.y_pred is None:
            self.predict()
        
        r2 = r2_score(self.y_test, self.y_pred)

        plt.figure(figsize=(8,6))
        plt.hist(r2, num_bins, edgecolor = 'black')
        plt.xlabel('R2_Score')
        plt.ylabel('Frequency')
        plt.title('R2 histogram')
        plt.show()

    def display_metrics(self):

        if self.y_pred is None:
            self.predict()

        r2 = r2_score(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)

        print('R2 Score: ', r2)
        print('Mean Squared Error: ', mse)
        print('Mean Absolute Error: ', mae)

