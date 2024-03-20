##########################################################################
#### Model train and evaluate utilities
#### Author : Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import numpy as np
import configparser
import os
from typing import Union
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import RepeatedKFold, StratifiedKFold, KFold, cross_validate
import joblib

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
    


class EarlyStopping:

    def __init__(self, metric, patience = 5, tol = 0.001, verbose = True):

        self.__counter = 0
        self.__stop = False
        self.verbose = verbose
        self.patience = patience
        self.tol = tol

        # cv specific attributes
        self.score = metric

        # tracking best model overall and each kfold run performance: containing score, k_iteration, model
        self.__best_score = None 
        self.__best_k = 0
        self.__best_model = None

    @property
    def stop(self):
        return self.__stop
    
    def print_verbose(self, message):
        if self.verbose:
            print(message)

    def get_best_fold(self) -> list:
        return [self.__best_score, self.__best_k, self.__best_model]

    # Call early stopping algorithm
    def __call__(self, current: list, mode = 'min'):

        # Update best results based on current fold results
        self.__update_best_score(current, mode)

        current_results, current_k, _ = current
        current_score = current_results[self.score]['mean']

        # Calculate the absolute change between current score and best score
        abs_change = abs(current_score - self.__best_score)

        # Check if there is no improvement but change is within tolerance
        if abs_change != 0 and abs_change < self.tol * self.__best_score:
            self.print_verbose("No improvement, but change is within tolerance.")
            
        # if no improvement exists outside of tolerance, start counting towards the early stop
        if (mode == 'min' and current_score > self.__best_score + self.tol*self.__best_score) or \
           (mode == 'max' and current_score < self.__best_score - self.tol*self.tol*self.__best_score):
            self.__counter += 1

            self.print_verbose(f'Early stopping counter: {self.__counter} out of {self.patience}')

        # Reset early stopping counter if an improvement exists: current = best
        else:
            self.__counter = 0

        # If counter reaches patience, send stop signal
        if self.__counter >= self.patience:

            self.__stop = True

            self.print_verbose('-'*72)
            self.print_verbose(f'Stopping kfold sensitivity early stopping at fold {current_k}')

        
    # Evaluate kfold cv run performance vs. previous best kfold run to update best model so far
    def __update_best_score(self, current: list, mode = 'min'):
        
        if mode not in ['max', 'min']:
            raise ValueError('Unsupported update mode. Supported modes: "min", "max"')
        
        current_results, current_k, current_model = current
        current_score = current_results[self.score]['mean']

        previous_best = self.__best_score
        
        # Update best attributes if they are none or if an improvement is seen
        if self.__best_score is None or \
           (mode == 'min' and current_score < self.__best_score) or \
           (mode == 'max' and current_score > self.__best_score):
            
            self.__best_score = current_score
            self.__best_k = current_k
            self.__best_model = current_model

            self.print_verbose(f'Best scores updated at fold {current_k}: {self.score} now at {self.__best_score} from {previous_best}')
                
class KFoldCrossValidator(PathConfig):

    def __init__(self, model, name: str, k_sens = True, verbose = True):

        super().__init__()

        self.model = model
        self.model_name = name
        self.k_sens = k_sens
        self.verbose = verbose
    
    def print_verbose(self, message):
        if self.verbose:
            print(message)

    def __call__(self, *args: Union[np.any, str], **kwargs: Union[np.any, str]) -> dict:
        
        kfolders = {'kfold': KFold,
                      'repeated': lambda n_splits: RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats), # If cv_type is 'repeated', specify the number of repeats
                      'stratified': StratifiedKFold}
        
        X, y, mode = args[0], args[1], args[2]

        # mode whether going for sknative or MLP kfold function
        if mode == 'sk_native':
            cv_fun = self.sk_native_cv
        elif mode == 'mlp':
            cv_fun = self.mlp_cv
        else:
            raise ValueError(f'Cross validator mode : {mode} not supported. Options are sk_native or mlp')

        # Optional kwargs depending on kfoldcv call
        cv_type = kwargs.get('cv_type')
        n_repeats = kwargs.get('repeats', 5)
        es_score = kwargs.get('earlystop_score', 'mse')
         
        # Select Validator object based on input arguments
        if cv_type in kfolders.keys():
            self.cv_class = kfolders[cv_type]
        else:
            raise ValueError('Kfold cross validator type not specified or not supported, cv_type = str must be included in cross_validator call')
            
        # Initialize early stopping logic
        early_stopper = EarlyStopping(es_score, patience= 5, tol= 0.005, verbose = True)

        if self.k_sens:
            min_k = kwargs.get('min_k',3)
            max_k = kwargs.get('max_k', 50)

            kf_cv_summary = self.ksens_loop(X, y, cv_fun, early_stopper, min_k, max_k)

        return kf_cv_summary   

    def ksens_loop(self, X, y, cv_fun, early_stopper, min_k, max_k):
        
        # number of kfolds to try as hyperparameter for the cross validation sensitivity
        folds = range(min_k,max_k+1)

        # List to store all metrics per kfold cross validation run
        cv_results = {f'kfold_{fold}':{} for fold in folds}
        
        for k in folds:

            self.print_verbose('-'*72)
            self.print_verbose(f'Starting Cross-Validation with {k} folds ...')
            
            # Container to store results per k-fold cv run
            fold_results = {}

            #Cross validation set splitter
            cv = self.cv_class(n_splits = k)

            scores = cv_fun(X, y, cv)

            # Store the overall metrics obtained for the k cross validation run tested
            for metric in scores.keys():
                fold_results[metric] = {'mean': np.mean(scores[metric]),
                                            'min': np.min(scores[metric]),
                                            'max': np.max(scores[metric])}
            
            # Update overall results with k-fold cv instance results
            cv_results[f'kfold_{k}'].update(fold_results)

            self.print_verbose('-'*72)
            for metric in fold_results.keys():
                self.print_verbose(f'Mean scores with {k} folds: {metric} = {fold_results[metric]["mean"]};' )
            self.print_verbose('-'*72)

            # Current kfold state
            current = [fold_results, k, self.model]

            # Early stopping for kfold sensitivity
            early_stopper(current)

            if early_stopper.stop: 
                break

        _, best_fold_idx, best_model = early_stopper.get_best_fold()

        # Drop all future kfolds after algorithm has decided to early stop at a best kfold
        folds_to_drop = [f'kfold_{k}' for k in range(best_fold_idx+1, max_k+1)]
        for key in folds_to_drop:
            del cv_results[key]
        
        # Save best model obtained
        joblib.dump(best_model,os.path.join(self.model_savepath, self.model_name, f'{self.model_name}_best_model_{best_fold_idx}_folds.pkl'))
        
        # Save metrics log for all kfold runs carried out
        with open(os.path.join(self.model_savepath, self.model_name, f'{self.model_name}_kfoldcv_scores.txt'), 'w') as file :
            for k, fold_run in enumerate(cv_results.keys()):
                file.write(f'Results for cv run with k={k+min_k}: {cv_results[fold_run]}' + '\n')
                file.write('-'*72 + '\n')
        
        # Returning final mean metrics for best fold
        kf_cv_summary = {'Best fold': best_fold_idx}

        for metric in cv_results[f'kfold_{best_fold_idx}']:
            kf_cv_summary[metric] = cv_results[f'kfold_{best_fold_idx}'][metric]['mean']
            
        return kf_cv_summary

    def sk_native_cv(self, X, y, cv):
            
        scores_abs = {}
        scores = {}

        rename_keys = {'test_r2': 'r2', 'test_neg_mean_absolute_error': 'mae',
                       'test_neg_mean_squared_error': 'mse', 'test_explained_variance': 'variance'}
        
        # Extract detailed scores and performance per model
        score_metrics = ['explained_variance','r2','neg_mean_squared_error','neg_mean_absolute_error']
        
        sk_scores = cross_validate(self.model, X, y, scoring=score_metrics,cv=cv, n_jobs=5, verbose=0) #number of folds X number of repeats

        # Store the overall metrics obtained for the k cross validation run tested
        for metric in sk_scores.keys():
            # Take absolute value from sklearn natively negative metrics
            if 'neg' in metric:
                scores_abs[metric] = np.abs(sk_scores[metric])
            else:
                scores_abs[metric] = sk_scores[metric]
        
        # rename metrics names for better readability in prints
        scores = {rename_keys.get(old_key): value for old_key, value in scores_abs.items()}
        
        return scores

    def mlp_cv(self,X, y):
        
        # number of kfolds to try as hyperparameter for the cross validation sensitivity
        folds = range(self.min_k,self.max_k+1)

        # List to store all metrics per kfold cross validation run
        cv_results = {f'kfold_{fold}':{} for fold in folds}
