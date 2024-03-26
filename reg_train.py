##########################################################################
#### Regression model train and deployment
#### Author : Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import pandas as pd
import configparser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from abc import ABC, abstractmethod
#Model regression imports
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
#Model metrics and utilities
from model_utils import KFoldCrossValidator, HyperParamTuning, ModelEvaluator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import R2Score
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
        if is_mlp:
            return tf.keras.models.load_model(model_dir)
        else:
            return joblib.load(model_dir)
    
    @staticmethod
    def get_cvargs(cv: str, score: str) -> dict:
        """
    Get arguments for cross-validator instance (kfolder or tuning).

    Parameters:
    - es_score (str): The earlystop score, for kfold instances.

    Returns:
    - Dictionary with cross-validation arguments.
    """
        args = {'kfold_early': {'cv_type': 'kfold',
                        'n_repeats': 3,
                        'min_k': 3,
                        'max_k':50,
                        'k': 5,
                        'earlystop_score': score},
                'hp_tuning': {'tuning_type': 'std',
                           'n_iter': 30,
                           'fit_score': score},
                'kfold_final': {'cv_type': 'kfold',
                        'n_repeats': 3,
                        'min_k': 3,
                        'max_k':50,
                        'k': 5,
                        'earlystop_score': score}}
        
        if cv not in args:
            raise TypeError(f'cross-validator specified {cv} does not match existing argument dictionaries')
        
        return args.get(cv)
        
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
    - 'min_k'm 'max_k': Specified when k_sens is True to determine the extent of the K values to test
    - 'k': Specified when kfold cv_type is specified, determining the number of folds to run

    Options for tuning_type:
    - 'std': Standard GridSearchCV.
    - 'halving': HalvingGridSearchCV.
    - 'random': RandomizedSearchCV. (optional) n_iter can be specified to change the number of random runs to be performed
    - 'halve_random': Combining Halving and Random search hyperparameter tuning.

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
        skip_kfold = cv_options.get('skip_kfold')
        k_sens = cv_options.get('ksens')
        skip_hp_tune = cv_options.get('hp_tune')

        # select features and args based on regressor type used
        native = 'mlp' if isinstance(self, MLP) else 'sk_native'
        es_score = 'loss' if isinstance(self, MLP) else 'mse'

        # skip or not early kfold
        if not skip_kfold:

            # kfold cross validator arguments
            cv_args = self.get_cvargs('kfold_early', es_score)
            
            # crossvalidator instance
            cross_validate = KFoldCrossValidator(model, model_name, native, k_sens = k_sens)

            cv_scores, model_dir = cross_validate(X_train_arr,y_train_arr, **cv_args)

            print(f'Summary scores from early {cv_args["cv_type"]} cross validation')
            print('-'*72)
            print(cv_scores)

            # Load cross validated model for further handling
            model = self.load_model(model_dir, isinstance(self,MLP))
        
        # skip or not hyperparam tuning
        if not skip_hp_tune:
        
            hyperparam_tuning = HyperParamTuning(model,model_name, native, verbose= True)

            # tuning arguments
            hptune_args = self.get_cvargs('hp_tuning', score='mse')

            # calling hyperparam tuning. if random selected
            tuned_model = hyperparam_tuning(X_train_arr, y_train_arr, **hptune_args)
            
            # Carry out further kfold with or w/o k sensitivity on tuned model
            further_kfold = input('Perform further kfold cross-validation (with sensitivity)? (y/ys/n): ')
            if 'y' in further_kfold.lower() :

                k_cv = {'ys': True, 'y': False}
                fcv_args = self.get_cvargs('kfold_final', es_score)

                further_kcv = KFoldCrossValidator(tuned_model, model_name, native, k_sens=k_cv.get(further_kfold.lower()))

                tuned_scores, tuned_model_dir = further_kcv(X_train_arr, y_train_arr, **fcv_args)

                print(f'Summary scores from {fcv_args["cv_type"]} cross validation after tuning')
                print('-'*72)
                print(tuned_scores)

                # load tuned and cross validated model to be returned
                tuned_model = self.load_model(tuned_model_dir,isinstance(self,MLP))
        
        # if hp tune was skipped but early kfold was run, return kfold model
        elif not skip_kfold and skip_hp_tune:
            return model
        
        # simply train the model
        else:
            tuned_model = self.fit_model(X_train_arr,y_train_arr,model)

        return tuned_model
    
    def model_evaluate(self,tuned_model,data_packs):
            
        model_eval = ModelEvaluator(tuned_model, data_packs)

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

    # Build net architecture without compiling, for later custom or sklearn pipeline handling
    def build_net(self):

        net = Sequential()
        
        #Hyperparams
        n_dense_layers = self.kwargs.get('n_dense', 2)
        n_shallow_layers = self.kwargs.get('n_shallow',2)
        n_nodes_dense = self.kwargs.get('n_nodes_d',128)
        n_nodes_shallow = self.kwargs.get('n_nodes_s', 64)
        act_fn = self.kwargs.get('act_fn', 'relu')
        lr = self.kwargs.get('lr',0.001)

        # Feature dimensions
        input_shape = self.kwargs.get('input_size',None)
        output_shape = self.kwargs.get('output_size', None)

        # Input layer
        net.add(InputLayer(shape=(input_shape,)))

        # Dense layers, with more nodes per layer
        for _ in range(n_dense_layers):
            net.add(Dense(n_nodes_dense,activation=act_fn))

        # Shallow layers, with less nodes per layer
        for _ in range(n_shallow_layers):
            net.add(Dense(n_nodes_shallow,activation=act_fn))

        # Output layer
        net.add(Dense(output_shape,activation=act_fn))

        # Network training utilities
        optimizer = Adam(learning_rate=lr)

        net.compile(optimizer= optimizer, loss = 'mean_squared_error', metrics=['mae', 'mse', R2Score()])

        return net

    def fit_model(self,X_train,y_train,model):

        epochs = self.kwargs.get('n_epochs', 1)
        batch_size = self.kwargs.get('batch_size', 1)

        stopper = EarlyStopping(monitor='val_loss', patience=10)
        
        scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        
        # Fit Keras native model
        model.fit(X_train,y_train,validation_split = 0.3,batch_size = batch_size, epochs=epochs, verbose=1, callbacks = [scheduler,stopper])

        return model

############################# REGRESSOR CHILD CLASSES ###################################
    
class DecisionTreeWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):

        #Hyperparams
        criterion = self.kwargs.get('criterion','squared_error')
        max_depth = self.kwargs.get('max_depth',None)
        min_samples_split = self.kwargs.get('min_samples_split',2)
        min_samples_leaf = self.kwargs.get('min_samples_leaf',1)
        min_impurity_decrease = self.kwargs.get('min_impurity_decrease',0)
        max_leaf_nodes = self.kwargs.get('max_leaf_nodes', None)
        splitter = self.kwargs.get('splitter', 'best')


        random_state = self.kwargs.get('random_state',2024)


        if max_depth is None:
            raise ValueError('Max_depth is required for Decision Tree Regressor')
        
        return DecisionTreeRegressor(criterion=criterion,
                                     max_depth=max_depth, min_samples_split=min_samples_split, 
                                     min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease,
                                     max_leaf_nodes= max_leaf_nodes, splitter= splitter,
                                     random_state = random_state)
    
    def model_build(self, data_packs, model, kfold, model_name):
        return super().model_build(data_packs, model, kfold, model_name)

class XGBoostWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):

        max_depth = self.kwargs.get('max_depth',None)
        n_estimators = self.kwargs.get('n_estimators',None)
        learning_rate = self.kwargs.get('learning_rate',0.1)
        min_child_weight = self.kwargs.get('min_child_weight',1 )
        subsample = self.kwargs.get('subsample',1 )
        colsample_bytree = self.kwargs.get('colsample_bytree',0.8)
        gamma = self.kwargs.get('gamma', 0.1)
        reg_lambda = self.kwargs.get('lambda', 0.01)
        reg_alpha = self.kwargs.get('alpha',0.01)

        random_state = self.kwargs.get('random_state',2024)

        if max_depth is None or n_estimators is None:
            raise ValueError('Missing input arguments: Max_depth/n_estimators')
        
        return XGBRegressor(max_depth=max_depth, n_estimators = n_estimators,
                            learning_rate = learning_rate, min_child_weight = min_child_weight,
                             subsample = subsample, colsample_bytree = colsample_bytree,
                              gamma = gamma, reg_lambda = reg_lambda, reg_alpha = reg_alpha,
                                random_state=random_state)
    
    def model_build(self, data_packs, model, kfold, model_name):
        return super().model_build(data_packs, model, kfold, model_name)
    
class RandomForestWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):

        n_estimators = self.kwargs.get('n_estimators',None)
        random_state = self.kwargs.get('random_state',2024)

        if n_estimators is None:
            raise ValueError('n_estimators is required for Random Forest Regressor')
        
        return RandomForestRegressor(n_estimators = n_estimators, random_state=random_state)
    
    def model_build(self, data_packs, model, kfold, model_name):
        return super().model_build(data_packs, model, kfold, model_name)
    
class SVMWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):
        
        c_coef = self.kwargs.get('C', None)
        epsilon = self.kwargs.get('epsilon',None)

        if c_coef is None or epsilon is None:
            raise ValueError(' C and epsilon required for SVM')

        return MultiOutputRegressor(SVR(C=c_coef,epsilon=epsilon))
    
    def model_build(self, data_packs, model, kfold, model_name):
        return super().model_build(data_packs, model, kfold, model_name)

class KNNWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):
        
        #Hyperparams
        n_neighbours = self.kwargs.get('n_neighbours', None)

        if n_neighbours is None:
            raise ValueError('n_neighbours required for KNN')
        
        return KNeighborsRegressor(n_neighbors=n_neighbours)
    
    def model_build(self, data_packs, model, kfold, model_name):
        return super().model_build(data_packs, model, kfold, model_name)

class MLPRegressorWrapper(MLP):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_model(self):
        
        # wrap Keras net as a sklearn regressor object
        net = self.build_net()

        return KerasRegressor(model = net,verbose=1)
    
    def model_build(self, data_packs, model, kfold, model_name):
        return super().model_build(data_packs, model, kfold, model_name)

class MLPWrapper(MLP):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):
        return self.build_net()
    
    def model_build(self, data_packs, model, kfold, model_name):
        return super().model_build(data_packs, model, kfold, model_name)     

def main():

    # Load data to process
    path = PathConfig()

    case = input('Select a study to process raw datasets (sp_(sv)geom, (sv)surf, (sv)geom): ')
    label_package = []
    data_packs = []

    # Read package names to later import
    with open(os.path.join(path.input_savepath,case,'Load_Labels.txt'), 'r') as file:
        lines = file.readlines()

        for line in lines:
            label_package.append(line.split('\n')[0])

    # Checking in PCA has been applied to the dataset
    if 'PCA_res' in label_package:
        pca = True
    else:
        pca = False
    
    # Save only train and test packs
    label_package =  [item for item in label_package if item not in ['full', 'PCA_res']]
    
    # Load pickle files
    for label in label_package:

        data_path = os.path.join(path.input_savepath,case,f'{label}.pkl')

        if os.path.exists(data_path):

            data_pack = pd.read_pickle(data_path)          
            data_packs.append(data_pack)
    
    ## Regressor instances, labels and hyperparameters
    
    model_names = {'dt': 'Decision_Tree', 
                    'xgb': 'XGBoost', 
                    'rf': 'Random_Forest',
                    'svm': 'Support_Vector_Machine',
                    'knn': 'K_Nearest_Neighbours',
                    'mlp_reg': 'MLP_Wrapped_Regressor',
                    'mlp': 'Multi_Layer_Perceptron'}
    
    wrapper_dict = {'dt': DecisionTreeWrapper, 
                    'xgb': XGBoostWrapper, 
                    'rf': RandomForestWrapper,
                    'svm': SVMWrapper,
                    'knn': KNNWrapper,
                    'mlp_reg': MLPRegressorWrapper,
                    'mlp': MLPWrapper}
    
    hyperparameters = {
        'dt': {'criterion': 'absolute_error',
                'max_depth': 8,
                'min_samples_split': 4,
                'min_samples_leaf': 4,
                'min_impurity_decrease': 0,
                'max_leaf_nodes': None,
                'splitter': 'best'}, 
        'xgb': {'max_depth': 1, 
                'n_estimators': 200, 
                'learning_rate': 0.3,
                'min_child_weight': 3, 
                'subsample': 1,
                'colsample_bytree': 1, 
                'gamma': 0,
                'lambda': 0.001, 
                'alpha': 0.05}, 
        'rf': {'n_estimators': 100},
        'svm': {'C': 1, 'epsilon': 0.1},
        'knn': {'n_neighbours': 10},
        'mlp_reg': {'n_dense' : 2,
                'n_shallow': 2,
                'n_nodes_d': 128,
                'n_nodes_s': 64,
                'n_epochs' : 100,
                'batch_size' : 1,
                'act_fn': 'relu',
                'input_size': data_packs[0].shape[-1],
                'output_size': data_packs[1].shape[-1]},
        'mlp': {'n_dense' : 2,
                'n_shallow': 2,
                'n_nodes_d': 128,
                'n_nodes_s': 64,
                'n_epochs' : 100,
                'batch_size' : 1,
                'act_fn': 'relu',
                'input_size': data_packs[0].shape[-1],
                'output_size': data_packs[1].shape[-1]}
                }
    
    # Model selection from user input
    model_choice = input('Select a regressor to train and deploy (dt, xgb, rf, svm, knn, mlp_reg, mlp): ')

    if model_choice not in wrapper_dict.keys():
        raise ValueError('Specified model is not supported')
    
    # selecting corresponding wrapper
    wrapper_model = wrapper_dict.get(model_choice)

    model_params = hyperparameters.get(model_choice)

    model_name = model_names.get(model_choice)

    skip_kfold = input('Skip pre-Kfold cross validation? (y/n): ')
    
    # Decide whether to do pre-kfold and include k sensitivity
    if skip_kfold.lower() == 'n':
        ksens = input('Include K-sensitivity? (y/n): ')
    else:
        ksens = 'n'
    
    skip_hp_tune = input('Skip hyperparameter tuning cross-validation? (y/n): ')

    cv_options = {'skip_kfold': True if skip_kfold.lower() == 'y' else False,
          'ksens' : True if ksens.lower() == 'y' else False,
          'hp_tune': True if skip_hp_tune.lower() == 'y' else False}

    # Instantiating the wrapper with the corresponding hyperparams
    model_instance = wrapper_model(**model_params)

    # Getting regressor object from wrapper
    model = model_instance.init_model()

    # Regression training and evaluation
    tuned_model = model_instance.model_train(data_packs, model, 
                                        cv_options, model_name)
    # Calling model evaluate with tuned model
    model_instance.model_evaluate(tuned_model, data_packs)

if __name__ == "__main__":
    main()