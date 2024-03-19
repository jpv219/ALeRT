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
    


class EarlyStopping:

    def __init__(self, metric, patience = 5, tol = 0.001, verbose = True):

        self.__counter = 0
        self.__stop = False
        self.verbose = verbose
        self.patience = patience
        self.tol = tol

        # cv specific attributes
        self.score = 'test_' + metric

        # tracking best model overall and each kfold run performance: containing score, k_iteration, model
        self.__best_score = None 
        self.__best_k = 0
        self.__best_model = None

    @property
    def stop(self):
        return self.__stop

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
            if self.verbose:
                print("No improvement, but change is within tolerance.")
            
        # if no improvement exists outside of tolerance, start counting towards the early stop
        if (mode == 'min' and current_score > self.__best_score + self.tol*self.__best_score) or \
           (mode == 'max' and current_score < self.__best_score - self.tol*self.tol*self.__best_score):
            self.__counter += 1

            if self.verbose:
                print(f'Early stopping counter: {self.__counter} out of {self.patience}')

        # Reset early stopping counter if an improvement exists: current = best
        else:
            self.__counter = 0

        # If counter reaches patience, send stop signal
        if self.__counter >= self.patience:

            self.__stop = True

            if self.verbose:
                print(f'Stopping kfold sensitivity early stopping at fold {current_k}')

        
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

            if self.verbose:
                print(f'Best scores updated at fold {current_k}: {self.score} now at {self.__best_score} from {previous_best}')
                

class KFoldCrossValidator(PathConfig):

    def __init__(self, cv_type, model,name, min_k=3, max_k=50, n_repeats = 5):

        super().__init__()

        self.model = model
        self.model_name = name
        self.min_k = min_k
        self.max_k = max_k

        validators = {'kfold': KFold,
                      'repeated': lambda n_splits: RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats), # If cv_type is 'repeated', specify the number of repeats
                      'stratified': StratifiedKFold}
                     
        # Select Validator object based on input arguments
        if cv_type in validators.keys():
            self.cv_class = validators[cv_type]

        else:
            raise ValueError('Kfold cross validator specified is not supported')

    def gen_kfold_cv(self, X, y):
        
        # number of kfolds to try as hyperparameter for the cross validation sensitivity
        folds = range(self.min_k,self.max_k+1)

        # List to store all metrics per kfold cross validation run
        cv_results = {f'kfold_{fold}':{} for fold in folds}
        
        # Initialize early stopping logic
        early_stopping = EarlyStopping('neg_mean_squared_error', patience= 5, tol= 0.005, verbose = True)

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
            current = [fold_results, k, self.model]

            # Early stopping for kfold sensitivity
            early_stopping(current)

            if early_stopping.stop: 
                break

        _, best_fold_idx, best_model = early_stopping.get_best_fold()

        # Drop all future kfolds after algorithm has decided to early stop at a best kfold
        folds_to_drop = [f'kfold_{k}' for k in range(best_fold_idx+1, self.max_k+1)]
        for key in folds_to_drop:
            del cv_results[key]
        
        # Save best model obtained
        joblib.dump(best_model,os.path.join(self.model_savepath, self.model_name, f'{self.model_name}_best_model_{best_fold_idx}_folds.pkl'))
        
        # Save metrics log for all kfold runs carried out
        with open(os.path.join(self.model_savepath, self.model_name, f'{self.model_name}_kfoldcv_scores.txt'), 'w') as file :
            for k, fold_run in enumerate(cv_results.keys()):
                file.write(f'Results for cv run with k={k+self.min_k}: {cv_results[fold_run]}' + '\n')
                file.write('-'*72 + '\n')
        
        # Returning final mean metrics for best fold
        kf_cv_summary = {'Best fold': best_fold_idx,
        'r2' : cv_results[f'kfold_{best_fold_idx}']['test_r2']['mean'],
        'mae' : cv_results[f'kfold_{best_fold_idx}']['test_neg_mean_absolute_error']['mean'],
        'mse' : cv_results[f'kfold_{best_fold_idx}']['test_neg_mean_squared_error']['mean'],
        'var' : cv_results[f'kfold_{best_fold_idx}']['test_explained_variance']['mean']
        }
            
        return kf_cv_summary


            