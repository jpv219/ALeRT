##########################################################################
#### Regression model train and deployment
#### Author : Juan Pablo Valdes
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

import numpy as np
import pandas as pd
import configparser
import os
from abc import ABC, abstractmethod
from matplotlib import pyplot
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_validate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
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

    def kfold_cv(self,X,y,model,label):

        # number of folds to try as hyperparameter for the cross validation
        kfolds=50
        folds = range(2,kfolds)

        # List to store all metrics per kfolds cross validates
        results = {f'fold_{fold}':{} for fold in folds}

        # Empty containers to store best model and track kfold sensitivity performance
        best_model = None
        best_score = float('-inf')  # Initialize with a very low value
        tolerance = 0.01

        for k in folds:

            # Container to store results per k-fold size iteration
            fold_results = {}

            #Cross validation set splitter
            cv = RepeatedKFold(n_splits=k, n_repeats=5)

            # Extract detailed scores and performance per model
            score_metrics = ['explained_variance','r2','neg_mean_squared_error','neg_mean_absolute_error']
            
            scores = cross_validate(model, X, y, scoring=score_metrics,cv=cv, n_jobs=4,verbose=1) #number of folds X number of repeats

            # Store the overall metrics obtained for the k-fold cross validation instance tested
            for metric in scores.keys():
                fold_results[metric] = {'mean' : np.mean(scores[metric]),
                                        'min': scores[metric].min(),
                                        'max' : scores[metric].max()}
                
            # Update overall results with k-fold instance results
            results[f'fold_{k}'].update(fold_results)
        
            # Evaluate kfold performance vs. previous k-size instace once to update best model so far
            if fold_results['test_neg_mean_squared_error']['mean'] > best_score:
                best_score = fold_results['test_neg_mean_squared_error']['mean']
                best_model = model

            # Check if MSE has stopped improving and stop the loop
            stop_crit = abs((results[f'fold_{k}']['test_neg_mean_squared_error']['mean'] - 
                             results[f'fold_{k-1}']['test_neg_mean_squared_error']['mean'])
                             /results[f'fold_{k}']['test_neg_mean_squared_error']['mean'] 
                            if k>2 else 0)
            
            if k>2 and stop_crit<tolerance:
                folds_to_drop = [f'fold_{k}' for k in range(k+1,kfolds)]
                final_results = {key: value for key,value in results.items() if key not in folds_to_drop}
                break

            else:
                final_results = results
        
        # Save best model obtained
        joblib.dump(best_model,os.path.join(self.model_savepath, f'{label}_best_kfold_model.pkl'))
        
        return final_results
        
    def model_eval(self, **kwargs):
        self.kwargs = kwargs

        pca = self.kwargs.get('PCA')
        model = self.kwargs.get('model')
        kfold = self.kwargs.get('kfold')
        label = self.kwargs.get('label')

        # Reading data arrays for model fit and eval
        data_packs = self.kwargs.get('data_packs',None)
        X_train, y_train, X_test, y_test = data_packs[:4]

        # Converting into numpy for model fit and prediction
        X_train_arr = X_train.to_numpy()
        y_train_arr = y_train.to_numpy()
        X_test_arr = X_test.to_numpy()
        y_test_arr = y_test.to_numpy()

        # Carry out repeated Kfold cross validation only on train sets
        if kfold.lower() == 'y':

            scores = self.kfold_cv(X_train_arr,y_train_arr,model,label)

        else:

            if isinstance(self,MLP):
            
                # Add MLP specific hyperparameters
                epochs = self.kwargs.get('n_epochs', 1)
                batch_size = self.kwargs.get('batch_size', 1)
                
                # Call model fit function
                tr_model = self.fit_model(X_train_arr,y_train_arr,model,epochs = epochs, batch_size = batch_size)

            else:

                # Call model fit function
                tr_model = self.fit_model(X_train_arr,y_train_arr,model)

            # Carry out predictions and evaluate model performance
            y_pred = tr_model.predict(X_test_arr)

            r2 = r2_score(y_test_arr,y_pred)
            mae = mean_absolute_error(y_test_arr,y_pred)
            mse = mean_squared_error(y_test_arr,y_pred)

            scores = [r2,mae,mse]

        return scores
    
    def fit_model(self,X_train,y_train,model,**kwargs):

        # Fit model from native sklearn wrapper and return trained model
        model.fit(X_train,y_train)

        return model

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

    def fit_model(self,X_train,y_train,model,**kwargs):

        epochs = self.kwargs.get('n_epochs', 1)
        batch_size = self.kwargs.get('batch_size', 1)

        scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        
        # Fit Keras native model
        model.fit(X_train,y_train,validation_split = 0.3,batch_size = batch_size, epochs=epochs, verbose=1, callbacks = [scheduler])

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
        random_state = self.kwargs.get('random_state',2024)


        if max_depth is None:
            raise ValueError('Max_depth is required for Decision Tree Regressor')
        
        return DecisionTreeRegressor(criterion=criterion,
                                     max_depth=max_depth, min_samples_split=min_samples_split, 
                                     min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease,
                                     random_state = random_state)
    
    def model_eval(self, **kwargs):
        return super().model_eval(**kwargs)

class XGBoostWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):

        max_depth = self.kwargs.get('max_depth',None)
        n_estimators = self.kwargs.get('n_estimators',None)
        random_state = self.kwargs.get('random_state',2024)

        if max_depth is None or n_estimators is None:
            raise ValueError('Missing input arguments: Max_depth/n_estimators')
        
        return XGBRegressor(max_depth=max_depth, n_estimators = n_estimators , random_state=random_state)
    
    def model_eval(self, **kwargs):
        return super().model_eval(**kwargs)
    
class RandomForestWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):

        n_estimators = self.kwargs.get('n_estimators',None)
        random_state = self.kwargs.get('random_state',2024)

        if n_estimators is None:
            raise ValueError('n_estimators is required for Random Forest Regressor')
        
        return RandomForestRegressor(n_estimators = n_estimators, random_state=random_state)
    
    def model_eval(self, **kwargs):
        return super().model_eval(**kwargs)
    
class SVMWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):
        
        c_coef = self.kwargs.get('C', None)
        epsilon = self.kwargs.get('epsilon',None)

        if c_coef is None or epsilon is None:
            raise ValueError(' C and epsilon required for SVM')

        return MultiOutputRegressor(SVR(C=c_coef,epsilon=epsilon))
    
    def model_eval(self, **kwargs):
        return super().model_eval(**kwargs)

class KNNWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):
        
        #Hyperparams
        n_neighbours = self.kwargs.get('n_neighbours', None)

        if n_neighbours is None:
            raise ValueError('n_neighbours required for KNN')
        
        return KNeighborsRegressor(n_neighbors=n_neighbours)
    
    def model_eval(self, **kwargs):
        return super().model_eval(**kwargs)

class MLPRegressorWrapper(MLP):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_model(self):
        
        # wrap Keras net as a sklearn regressor object
        net = self.build_net()

        return KerasRegressor(model = net,verbose=1)
    
    def model_eval(self, **kwargs):
        return super().model_eval(**kwargs)

class MLPWrapper(MLP):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):
        return self.build_net()
    
    def model_eval(self, **kwargs):
        return super().model_eval(**kwargs)
    
    def kfold_cv(self,X,y,net,label):

        # number of folds to try as hyperparameter for the cross validation
        kfolds=4
        folds = range(2,kfolds)

        # List to store all metrics per kfolds cross validates
        results = {f'fold_{fold}':{} for fold in folds}

        tolerance = 0.01

        for k in folds:

            kfold_results = {}
            fold_result = {}

            latest_checkpoint = None
            
            checkpoint_dir = os.path.join(self.model_savepath, f'{label}_{k}_folds')

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            else:
                for filename in os.listdir(checkpoint_dir):
                    file_path = os.path.join(checkpoint_dir,filename)
                    os.remove(file_path)
        
            cv = RepeatedKFold(n_splits=k, n_repeats=5)

            print('-'*72)
            print(f'Starting Cross-Validation with {k} folds ...')

            best_val_loss = float('inf')

            for repeat_idx, (train, test) in enumerate(cv.split(X,y)):
                
                # Creating callbacks
                checkpoint_path = os.path.join(checkpoint_dir, f'repeat_{repeat_idx}_best.keras')
                checkpoint = ModelCheckpoint(checkpoint_path, 
                                    monitor='val_loss', save_best_only= True, 
                                verbose=1, mode='min', initial_value_threshold = best_val_loss)
                
                callbacks_list = [checkpoint]

                history = net.fit(X[train],y[train], 
                                validation_data=(X[test], y[test]), epochs = 5, batch_size = 1,
                                callbacks=callbacks_list,verbose=0)
                
                if os.path.exists(checkpoint_path):

                    latest_checkpoint = checkpoint_path
                
                net.load_weights(latest_checkpoint)

                scores = net.evaluate(X[test], y[test], verbose=0)

                if scores[0]< best_val_loss:
                    best_val_loss = scores[0]
                    
                for i, metric in enumerate(history.history.keys()):
                    if i < len(scores):
                        if metric in fold_result:
                            fold_result[metric].append(scores[i])
                        else:
                            fold_result[metric] = [scores[i]]
                    else:
                        break

                tf.keras.backend.clear_session()
            
            # Store the metrics obtained for each kfold cross validation
            for metric in fold_result.keys():
                kfold_results[metric] = {'mean' : np.mean(fold_result[metric]),
                                        'min': min(fold_result[metric]),
                                        'max' : max(fold_result[metric])}
                
            results[f'fold_{k}'].update(kfold_results)

            print('-'*72)
            for metric in fold_result.keys():
                print(f'Mean scores with {k} kfolds: {metric} = {kfold_results[metric]["mean"]};' )
            
            loss_score = next(iter(fold_result.keys()))
            
            # Check if MSE has stopped improving and stop the loop
            stop_crit = abs((results[f'fold_{k}'][loss_score]['mean'] - 
                            results[f'fold_{k-1}'][loss_score]['mean'])
                            /results[f'fold_{k}'][loss_score]['mean'] 
                                if k>2 else 0)
            
            if k>2 and stop_crit<tolerance:
                folds_to_drop = [f'fold_{k}' for k in range(k+1,kfolds)]
                final_results = {key: value for key,value in results.items() if key not in folds_to_drop}
                break

            else:
                final_results = results  

        return final_results


def main():

    # Load data to process
    path = PathConfig()

    case = 'sp_geom'
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
    
    model_labels = {'dt': 'Decision_Tree', 
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
        'dt': {'criterion': 'squared_error',
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'min_impurity_decrease': 0}, 
        'xgb': {'max_depth': 5, 'n_estimators': 100}, 
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

    # selecting corresponding wrapper
    wrapper_model = wrapper_dict.get(model_choice)

    model_params = hyperparameters.get(model_choice)

    model_label = model_labels.get(model_choice)

    kfold_choice = input('Carry out K-fold cross validation? (y/n): ')

    # Instantiating the wrapper with the corresponding hyperparams
    model_instance = wrapper_model(**model_params)

    # Getting regressor object from wrapper
    model = model_instance.init_model()

    # Regression training and evaluation
    scores = model_instance.model_eval(data_packs = data_packs, model=model, 
                                       PCA=pca, kfold = kfold_choice,label = model_label,**model_params)
    
    print(scores)
 

if __name__ == "__main__":
    main()