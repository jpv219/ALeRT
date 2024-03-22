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
#Model metrics and utilities
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from model_utils import KFoldCrossValidator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import R2Score

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

    def fit_model(self,X_train,y_train,model):

        # Fit model from native sklearn wrapper and return trained model
        model.fit(X_train,y_train)

        return model
        
    # Model build main pipeline: kfold + gridsearch + kfold
    def model_build(self, data_packs, model, ksens, model_name):

        # Reading data arrays for model fit and eval
        X_train, y_train, X_test, y_test = data_packs[:4]

        # Converting into numpy for model fit and prediction
        X_train_arr = X_train.to_numpy()
        y_train_arr = y_train.to_numpy()
        X_test_arr = X_test.to_numpy()
        y_test_arr = y_test.to_numpy()

        # select arguments based on regressor type used
        if isinstance(self,MLP):
            native = 'mlp'
            es_score = 'loss'
            
        else:
            native = 'sk_native'
            es_score = 'mse'

        # Determine whether to carry out k sensitivity study on first kfold loop
        if ksens.lower() == 'y':
            k_sens = True
        else:
            k_sens = False            

        # cross validator arguments
        cv_args = {'cv_type': 'kfold',
                    'n_repeats': 1,
                    'min_k': 3,
                    'max_k':50,
                    'k': 8,
                    'earlystop_score': es_score}
        
        # crossvalidator instance
        cross_validate = KFoldCrossValidator(model, model_name, native, k_sens = k_sens)

        cv_scores, model_dir = cross_validate(X_train_arr,y_train_arr, **cv_args)

        print(model_dir)

        # select arguments based on regressor type used
        if isinstance(self,MLP):
            model = tf.keras.models.load_model(model_dir)
            
        else:
            model = joblib.load(model_dir)
        

        return cv_scores

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
        random_state = self.kwargs.get('random_state',2024)


        if max_depth is None:
            raise ValueError('Max_depth is required for Decision Tree Regressor')
        
        return DecisionTreeRegressor(criterion=criterion,
                                     max_depth=max_depth, min_samples_split=min_samples_split, 
                                     min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease,
                                     random_state = random_state)
    
    def model_build(self, data_packs, model, kfold, model_name):
        return super().model_build(data_packs, model, kfold, model_name)

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

    if model_choice not in wrapper_dict.keys():
        raise ValueError('Specified model is not supported')
    
    # selecting corresponding wrapper
    wrapper_model = wrapper_dict.get(model_choice)

    model_params = hyperparameters.get(model_choice)

    model_name = model_names.get(model_choice)

    ksens = input('Carry out Kfold sensitivity cross validation? (y/n): ')

    # Instantiating the wrapper with the corresponding hyperparams
    model_instance = wrapper_model(**model_params)

    # Getting regressor object from wrapper
    model = model_instance.init_model()

    # Regression training and evaluation
    scores = model_instance.model_build(data_packs, model, 
                                        ksens, model_name)
    
    print(scores)
 

if __name__ == "__main__":
    main()