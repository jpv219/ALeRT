##########################################################################
#### Regressor base classes
#### Author: Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

from abc import ABC, abstractmethod
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import os
#Model metrics and utilities
from model_utils import KFoldCrossValidator, HyperParamTuning, ModelEvaluator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import R2Score
import joblib
from paths import PathConfig

############################# ABSTRACT PARENT CLASS ######################################

class Regressor(ABC,PathConfig):
    """Abstract base class for regression models."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    @abstractmethod
    def init_model(self):
        """Initialize the regression model."""
        pass

    @staticmethod
    def fit_model(X_train,y_train,model):

        # Fit model from native sklearn wrapper and return trained model
        model.fit(X_train,y_train)

        return model
    
    @staticmethod
    def load_model(model_dir: str, is_mlp: bool):
        """
    Load the model based on whether it's an MLP model or not.

    Parameters:
    - model_dir (str): Directory where the model is stored.
    - is_mlp (bool): Flag indicating whether the model is an MLP model or not.

    Returns:
    - Loaded model object.
    """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model state at '{model_dir}' does not exist.")
        
        if is_mlp:
            return tf.keras.models.load_model(model_dir)
        else:
            return joblib.load(model_dir)

    @staticmethod
    def save_model(model, model_dir: str, is_mlp: bool):
        """
        Save the model based on whether it's an MLP model or not.

        Parameters:
        - model: model state to save
        - model_dir (str): Directory where the model will be saved.
        - is_mlp (bool): Flag indicating whether the model is an MLP model or not.

        """
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")
        
        if is_mlp:
            model.save(os.path.join(model_dir, 'best_model.keras'))
        else:
            joblib.dump(model, os.path.join(model_dir, 'best_model.pkl'))

    @staticmethod
    def get_cvargs(cv: str, score: str) -> dict:
        """
    Get arguments for cross-validator instance (kfolder or tuning).

    Parameters:
    - es_score (str): The earlystop score, for kfold instances.

    Returns:
    - Dictionary with cross-validation arguments.
    """
        args = {'early_kf': {'cv_type': 'kfold',
                        'n_repeats': 3,
                        'min_k': 3,
                        'max_k':50,
                        'k': 5,
                        'earlystop_score': score},
                'hp_tuning': {'sk_tuning_type': 'std',
                              'mlp_tuning_type': 'bayesian',
                           'n_iter': 30,
                           'max_trials': 10,
                           'fit_score': score},
                'final_kf': {'cv_type': 'kfold',
                        'n_repeats': 3,
                        'min_k': 3,
                        'max_k':50,
                        'k': 5,
                        'earlystop_score': score}}
        
        if cv not in args:
            raise TypeError(f'cross-validator specified {cv} does not match existing argument dictionaries')
        
        return args.get(cv)

    # Kfold cross validation operation from model utils
    def kfold_cv(self, X: np.array, y: np.array, model, model_name: str, native: str, es_score: str, k_sens: bool, step: str):

        # kfold cross validator arguments
        cv_args = self.get_cvargs(step, es_score)
        
        # crossvalidator instance
        cross_validate = KFoldCrossValidator(model, model_name, native, k_sens = k_sens)

        cv_scores, model_dir = cross_validate(X,y, **cv_args)

        print(f'Summary scores from {step} {cv_args["cv_type"]} cross validation')
        print('-'*72)
        print(cv_scores)

        # Load cross validated model for further handling
        cv_model = self.load_model(model_dir, isinstance(self,MLP))

        return cv_model

    # Model train main pipeline: kfold + gridsearch + kfold
    def model_train(self, data_packs: list, model, cv_options: dict, model_name: str):

        """
    Trains a regressor using K-fold cross-validation
    along with hyperparameter tuning.

    Parameters:
    - data_packs (list): Data arrays for model fitting and evaluation.
                          Typically includes features (X_train, X_test) and target values (y_train, y_test).
    - model: Regressor object to be trained, whether sklearn or MLP native
    - cv_choices (dict): User inputs to determine which steps to carry out during model building and train.
                - skip_kfold (bool): Whether to skip the K-fold cross-validation step.
                - ksens (bool): Whether to carry out sensitvity analysis on the numer of k-folds to perform, if applicable.
                - skip_hp_tune (bool): Whether to skip the Hyperparameter tuning cross-validation step.
    - model_name (str): Name or identifier for the model being trained.

    Options for es_score:
    - 'mse': Mean Squared Error
    - 'loss': Loss as defined for the MLP models (usually mse)
    - 'mae': Mean Absolute Error
    - 'r2': R-squared score

    Options for cv_type:
    - 'kfold': Standard Kfold cross validator
    - 'repeated': Repeat Kfold cross validator. n_repeats has to be specified alongside this cv

    Options in cv_args:
    - 'min_k' 'max_k': Specified when k_sens is True to determine the extent of the K values to test
    - 'k': Specified when kfold cv_type is specified, determining the number of folds to run

    Options for sk_tuning_type:
    - 'std': Standard GridSearchCV.
    - 'halving': HalvingGridSearchCV.
    - 'random': RandomizedSearchCV. (optional) n_iter can be specified to change the number of random runs to be performed
    - 'halve_random': Combining Halving and Random search hyperparameter tuning.

    Options for mlp_tuning_type:
    - 'hyperband': 
    - 'bayesian': 
    - 'random': 
    - 'grid_search': 

    Options for fit_score:
    - 'mse': Mean Squared Error
    - 'mae': Mean Absolute Error
    - 'r2': R-squared score

    Returns:
    - tuned_model: The trained model with optimized hyperparameters.
    """

        # Reading data arrays for model fit and eval
        X_train, y_train = data_packs[:2]

        # Converting into numpy for model fit and prediction
        X_train_arr = X_train.to_numpy()
        y_train_arr = y_train.to_numpy()

        # Kfold user inputs
        do_kfold = cv_options.get('do_kfold')
        k_sens = cv_options.get('ksens')
        do_hp_tune = cv_options.get('do_hp_tune')

        # select features and args based on regressor type used
        native = 'mlp' if isinstance(self, MLP) else 'sk_native'
        es_score = 'loss' if isinstance(self, MLP) else 'mse'

        # skip or not early kfold
        if do_kfold:

            model = self.kfold_cv(X_train_arr, y_train_arr, model, model_name,
                                      native, es_score, k_sens, step = 'early_kf')
        
        # skip or not hyperparam tuning
        if do_hp_tune:
        
            hyperparam_tuning = HyperParamTuning(model,model_name, native, verbose= True)

            # tuning arguments
            hptune_args = self.get_cvargs('hp_tuning', score=es_score)

            # update hp tune args if the instance is MLP and network architecture info is needed
            if isinstance(self,MLP):
                hptune_args['input_size'] = self.kwargs.get('input_size')
                hptune_args['output_size'] = self.kwargs.get('output_size')
                hptune_args['n_features'] = self.kwargs.get('n_features')

            # calling hyperparam tuning. if random selected
            tuned_model = hyperparam_tuning(X_train_arr, y_train_arr, **hptune_args)
            
            # Carry out further kfold with or w/o k sensitivity on tuned model
            further_kfold = input('Perform further kfold cross-validation (with sensitivity)? (y/ys/n): ')

            if 'y' in further_kfold.lower() :

                k_cv = {'ys': True, 'y': False}

                tuned_model = self.kfold_cv(X_train_arr,y_train_arr, tuned_model, model_name,
                                                native, es_score, k_sens=k_cv.get(further_kfold.lower()), 
                                                step = 'final_kf')
        
        # if hp tune was skipped but early kfold was run, return kfold model
        elif do_kfold and not do_hp_tune:
            return model
        
        # if kf cross validation wants to be executed on the mlp network despite not running hyperparam tuning
        elif isinstance(self,MLP) and not do_hp_tune:

            do_kf_mlp = input('Perform Kfold cross-validation (with sensitivity)? (y/ys/n): ')

            if 'y' in do_kf_mlp.lower():

                k_cv = {'ys': True, 'y': False}

                model = self.kfold_cv(X_train_arr,y_train_arr, model, model_name,
                                native, es_score, k_sens=k_cv.get(do_kf_mlp.lower()), 
                                step = 'final_kf')
            
            else:
                model = self.fit_model(X_train_arr,y_train_arr, model)
            
            return model

        # simply train the model
        else:
            tuned_model = self.fit_model(X_train_arr,y_train_arr,model)

        return tuned_model
    
    def model_evaluate(self,tuned_model,data_packs,case,pca, datasample):
            
        model_eval = ModelEvaluator(tuned_model, data_packs,case,pca, datasample)

        print('-'*72)
        print('Evaluating final model performance')
        print('-'*72)

        model_eval.plot_dispersion()
        model_eval.plot_r2_hist()
        model_eval.display_metrics()

############################# MLP PARENT CLASS ###########################################

class MLP(Regressor):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def init_model(self):
        """Initialize the regression model."""
        pass

    @abstractmethod
    def get_network(self):
        """Construct and return network architecture."""
        pass

    # Build net architecture without compiling, for later custom pipeline handling
    def build_net(self):

        net = self.get_network()
        
        lr = self.kwargs.get('lr',0.001)

        # Network training utilities
        optimizer = Adam(learning_rate=lr)

        net.compile(optimizer= optimizer, loss = 'mean_squared_error', metrics=['mae', 'mse', R2Score()])

        return net

    def fit_model(self,X_train,y_train,model):

        epochs = self.kwargs.get('n_epochs', 1)
        batch_size = self.kwargs.get('batch_size', 1)

        stopper = EarlyStopping(monitor='val_loss', patience=10)
        
        scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        
        # Fit Keras native model
        model.fit(X_train,y_train,validation_split = 0.3,batch_size = batch_size, epochs=epochs, verbose=0, callbacks = [scheduler,stopper])

        return model
