##########################################################################
#### Regression model library
#### Author: Juan Pablo Valdes and Fuyue Liang
### First commit: Feb 2024
### Department of Chemical Engineering, Imperial College London
##########################################################################

from base_model import Regressor, MLP
#Model regression imports
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Sequential, Model
from keras.layers import InputLayer, Dense, Input, Concatenate, Reshape

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
    
class RandomForestWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):

        n_estimators = self.kwargs.get('n_estimators',None)
        max_depth = self.kwargs.get('max_depth')
        max_features = self.kwargs.get('max_features')
        min_samples_split = self.kwargs.get('min_samples_split')
        min_samples_leaf = self.kwargs.get('min_samples_leaf')
        bootstrap = self.kwargs.get('bootstrap')
        random_state = self.kwargs.get('random_state',2024)

        if n_estimators is None:
            raise ValueError('n_estimators is required for Random Forest Regressor')
        
        return RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth,
                                     max_features=max_features,min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                       bootstrap=bootstrap, random_state=random_state)

class SVMWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):
        
        estimator__c = self.kwargs.get('C', None)
        estimator__epsilon = self.kwargs.get('epsilon',None)
        estimator__kernel = self.kwargs.get('kernel')
        estimator__gamma = self.kwargs.get('gamma')
        estimator__degree = self.kwargs.get('degree')
        estimator__coef0 = self.kwargs.get('coef0')

        if estimator__c is None or estimator__epsilon is None:
            raise ValueError('C and epsilon required for SVM')

        return MultiOutputRegressor(SVR(kernel=estimator__kernel,degree=estimator__degree,
                                        gamma= estimator__gamma, coef0= estimator__coef0, C=estimator__c,
                                        epsilon= estimator__epsilon))
    
class KNNWrapper(Regressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):
        
        #Hyperparams
        n_neighbors = self.kwargs.get('n_neighbors', None)
        weights = self.kwargs.get('weights')
        p = self.kwargs.get('p')
        algorithm = self.kwargs.get('algorithm')
        leaf_size = self.kwargs.get('leaf_size')
        metric = self.kwargs.get('metric')

        if n_neighbors is None:
            raise ValueError('n_neighbors required for KNN')
        
        return KNeighborsRegressor(n_neighbors=n_neighbors, weights= weights, p=p,
                                   algorithm= algorithm, leaf_size= leaf_size, metric= metric)
    
class MLPBranchedWrapper(MLP):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_model(self):
        return self.build_net()
    
    def get_network(self):

        #Hyperparams
        n_nodes_1 = self.kwargs.get('n_nodes_1',64)
        n_nodes_2 = self.kwargs.get('n_nodes_2', 32)
        n_nodes_br = self.kwargs.get('n_nodes_br', 32)
        act_fn = self.kwargs.get('act_fn', 'relu')

        # Feature dimensions
        input_shape = self.kwargs.get('input_size',None)
        output_shape = self.kwargs.get('output_size', None)
        n_features = self.kwargs.get('n_features')
        
        inputs = Input(shape=(input_shape,))

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

        reshaped_out = Reshape((output_shape,))(concatenated)

        net = Model(inputs = inputs, outputs = reshaped_out)

        return net
  
class MLPWrapper(MLP):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):
        return self.build_net()
    
    def get_network(self):
        
        net = Sequential()
        
        #Hyperparams
        n_dense_layers = self.kwargs.get('n_dense', 2)
        n_shallow_layers = self.kwargs.get('n_shallow',2)
        n_nodes_dense = self.kwargs.get('n_nodes_d',128)
        n_nodes_shallow = self.kwargs.get('n_nodes_s', 64)
        act_fn = self.kwargs.get('act_fn', 'relu')

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
        net.add(Dense(output_shape,activation='linear'))

        return net
    
############################# MODEL CONFIG ###################################

class ModelConfig:
        
    ## Regressor instances, labels and hyperparameters
    model_names = {'dt': 'Decision_Tree', 
                    'xgb': 'XGBoost', 
                    'rf': 'Random_Forest',
                    'svm': 'Support_Vector_Machine',
                    'knn': 'K_Nearest_Neighbours',
                    'mlp_br': 'MLP_Branched_Network',
                    'mlp': 'Multi_Layer_Perceptron'}
    
    wrapper_dict = {
        'dt': DecisionTreeWrapper, 
        'xgb': XGBoostWrapper, 
        'rf': RandomForestWrapper,
        'svm': SVMWrapper,
        'knn': KNNWrapper,
        'mlp_br': MLPBranchedWrapper,
        'mlp': MLPWrapper
    }

    hyperparameters = {
        'dt': {'criterion': 'squared_error',
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'min_impurity_decrease': 0,
                'max_leaf_nodes': None,
                'splitter': 'best'}, 
        'xgb': {'max_depth': 1, 
                'n_estimators': 150, 
                'learning_rate': 0.1,
                'min_child_weight': 3, 
                'subsample': 1,
                'colsample_bytree': 1, 
                'gamma': 0.01,
                'lambda': 0.001, 
                'alpha': 0.05}, 
        'rf': {'n_estimators': 100,
               'max_depth': 10,
               'max_features': 1,
               'min_samples_split': 2,
               'min_samples_leaf': 1,
               'bootstrap': True
               },
        'svm': {'C': 1, 
                'epsilon': 0.1,
                'kernel': 'rbf',
                'gamma': 'scale',
                'degree': 3,
                'coef0': 0},
        'knn': {'n_neighbors': 10,
                'weights': 'uniform',
                'p': 2,
                'algorithm': 'auto',
                'leaf_size': 30,
                'metric': 'minkowski'},
        'mlp_br': {'n_nodes_1' : 128,
                    'n_nodes_2': 64,
                    'n_nodes_br': 32,
                    'n_epochs' : 500,
                    'batch_size' : 1,
                    'act_fn': 'relu',
                    'lr': 0.01},
        'mlp': {'n_dense' : 4,
                'n_shallow': 2,
                'n_nodes_d': 128,
                'n_nodes_s': 64,
                'n_epochs' : 500,
                'batch_size' : 1,
                'act_fn': 'relu',
                'lr': 0.001}
    }

    @classmethod
    def get_wrapper(cls, key):
        if key not in cls.wrapper_dict.keys():
            raise ValueError(f'Specified model {key} is not supported')
        
        return cls.wrapper_dict.get(key)
    
    @classmethod
    def get_hyperparameters(cls, key):
        return cls.hyperparameters.get(key)
    
    @classmethod
    def get_model_name(cls, key):
        return cls.model_names.get(key)