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
from typing import Union
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import RepeatedKFold, KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras.callbacks import ModelCheckpoint, EarlyStopping
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

    def __init__(self, model, name: str, native: str, k_sens = True, verbose = True):

        super().__init__()

        model_abbr = {'Decision_Tree':'dt', 
                    'XGBoost':'xgb', 
                    'Random_Forest': 'rf',
                    'Support_Vector_Machine': 'rf',
                    'K_Nearest_Neighbours': 'knn',
                    'MLP_Wrapped_Regressor': 'mlp_reg',
                    'Multi_Layer_Perceptron': 'mlp'}
        
        self.model_name = name
        self.model_abbr = model_abbr.get(name)
        
        self.model = model
        self.k_sens = k_sens
        self.verbose = verbose
        self.native = native

        self.chk_dir = os.path.join(self.model_savepath, self.model_name)

        self.bestmodel_path = ''
    
    def __call__(self, *args: Union[np.any, str], **kwargs: Union[np.any, str]) -> dict:
               
        X, y = args[0], args[1]

        # Optional kwargs depending on kfoldcv call
        cv_type = kwargs.get('cv_type')
        n_repeats = kwargs.get('n_repeats', 5)
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
                       'test_neg_mean_squared_error': 'mse', 'test_explained_variance': 'variance'}
        
        # Extract detailed scores and performance per model
        score_metrics = ['explained_variance','r2','neg_mean_squared_error','neg_mean_absolute_error']
        
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




########################### MODEL EVALUATION ##############################################
                    
class ModelEvaluator(PathConfig):

    def __init__(self, model, x_test_df: pd.DataFrame, y_test_df: pd.DataFrame, native: str):
        super().__init__()

        self.model = model
        self.x_test_df = x_test_df
        self.y_test_df = y_test_df

        self.x_test = x_test_df.to_numpy()
        self.y_test = y_test_df.to_numpy()

        self.y_pred = None
        self.native = native

    def predict(self):
        self.y_pred = self.model.predict(self.x_test)

    def plot_dispersion(self):

        if self.y_pred is None:
            self.predict()

        plt.figure(figsize=(8,6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.5)
        plt.xlabel('True Data')
        plt.ylabel('Predicted Data')
        plt.title('True vs. Pred dispersion plot')
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

