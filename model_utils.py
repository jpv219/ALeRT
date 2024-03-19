##########################################################################
#### Model train and evaluate utilities
#### Author : Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import numpy as np
import configparser
import os
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
    


class EarlyStopping(PathConfig):

    def __init__(self):
        super().__init__()

    @staticmethod
    def update_best_score(current: list, best: list, mode = 'min') -> list:
        
        current_score, current_k, current_model = current
        best_score, best_k, best_model = best
        
        # Target objective is to minimize the measured metric/score
        if mode == 'min':
            if current_score < best_score:
                best_score = current_score
                best_k = current_k
                best_model = current_model
            else:
                return current
            
        # Target objective is to maximize the measured metric/score
        elif mode == 'max':
            if current_score > best_score:
                best_score = current_score
                best_k = current_k
                best_model = current_model
            else:
                return current
        
        else:
            raise ValueError('Update mode specified not supported')

        
        print(f'Best scores updated at fold {current_k}')

        return [best_score, best_k, best_model]

    @staticmethod
    def check_early_stopping(cv_results, min_k, kfolds, k, metric, tol = 0.01, patience = 5, mode = 'mean') -> tuple[bool,dict]:
        
        score = 'test_' + metric
        stop = False
        patience_counter = 0

        # Check if metric has stopped improving from the previous kfold instance run and stop the loop
        stop_crit = abs((cv_results[f'kfold_{k}'][score][mode] - 
                             cv_results[f'kfold_{k-1}'][score][mode])
                             /cv_results[f'kfold_{k}'][score][mode] 
                            if k>min_k else 1)
        
        # Early stopping algorithm
        if k>min_k and stop_crit<tol:
            
            stop = True

            # Drop all future kfold runs if algorithm chooses to stop
            folds_to_drop = [f'kfold_{k}' for k in range(k+1,kfolds+1)]

            for key in folds_to_drop:
                del cv_results[key]

            print(f'Stopping kfold sensitivity early stopping at fold {k}')
        
        return stop, cv_results
        


class KFoldCrossValidator(EarlyStopping,PathConfig):

    def __init__(self, cv_type, model,name, n_repeats = 5):

        super().__init__()
        self.model = model
        self.model_name = name

        validators = {'kfold': KFold,
                      'repeated': lambda n_splits: RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats), # If cv_type is 'repeated', specify the number of repeats
                      'stratified': StratifiedKFold}
                
        
        if cv_type in validators.keys():
            self.cv_class = validators[cv_type]

        else:
            raise ValueError('Kfold cross validator specified is not supported')

    def gen_kfold_cv(self, X, y, min_k=3, kfolds=50):
        
        # number of kfolds to try as hyperparameter for the cross validation sensitivity
        folds = range(min_k,kfolds+1)

        # List to store all metrics per kfold cross validation run
        cv_results = {f'kfold_{fold}':{} for fold in folds}
        
        # Empty containers to track best model overall and each kfold run performance: containing score, k_iteration, model
        best_model = None
        best_score = float('inf')  # Initialize with a very large value
        best_fold_idx = 0
        best = [best_score, best_fold_idx, best_model]

        for k in folds:

            # Container to store results per k-fold cv run
            fold_results = {}

            #Cross validation set splitter
            cv = self.cv_class(n_splits = k)

            # Extract detailed scores and performance per model
            score_metrics = ['explained_variance','r2','neg_mean_squared_error','neg_mean_absolute_error']
            
            scores = cross_validate(self.model, X, y, scoring=score_metrics,cv=cv, n_jobs=5,verbose=0) #number of folds X number of repeats

            # Store the overall metrics obtained for the k cross validation run tested
            for metric in scores.keys():
                # Take absolute value from sklearn natively negative metrics
                if 'neg' in metric:
                    scores_abs = np.abs(scores[metric])
                else:
                    scores_abs = scores[metric]
                
                fold_results[metric] = {'mean' : np.mean(scores_abs),
                                            'min': np.min(scores_abs),
                                            'max' : np.max(scores_abs)}
            
            # Update overall results with k-fold cv instance results
            cv_results[f'kfold_{k}'].update(fold_results)

            # Current kfold state
            current = [fold_results['test_neg_mean_squared_error']['mean'], k, self.model]

            # Evaluate kfold cv run performance vs. previous best kfold run to update best model so far
            best = self.update_best_score(current, best, mode = 'min')

            # Early stopping condition
            stop, cv_results = self.check_early_stopping(cv_results, min_k, kfolds, k, metric = 'neg_mean_squared_error')

            if stop: 
                break

        best_fold_idx = best[1]
        best_model = best[2]
        
        # Save best model obtained
        joblib.dump(best_model,os.path.join(self.model_savepath, self.model_name, f'{self.model_name}_best_model_{best_fold_idx}_folds.pkl'))
        
        # Save metrics log for all kfold runs carried out
        with open(os.path.join(self.model_savepath, self.model_name, f'{self.model_name}_kfoldcv_scores.txt'), 'w') as file :
            for k, fold_run in enumerate(cv_results.keys()):
                file.write(f'Results for cv run with k={k+min_k}: {cv_results[fold_run]}' + '\n')
                file.write('-'*72 + '\n')
        
        # Returning final mean metrics for best fold
        kf_cv_summary = {'Best fold': best_fold_idx,
        'r2' : cv_results[f'kfold_{best_fold_idx}']['test_r2']['mean'],
        'mae' : cv_results[f'kfold_{best_fold_idx}']['test_neg_mean_absolute_error']['mean'],
        'mse' : cv_results[f'kfold_{best_fold_idx}']['test_neg_mean_squared_error']['mean'],
        'var' : cv_results[f'kfold_{best_fold_idx}']['test_explained_variance']['mean']
        }
            
        return kf_cv_summary


            