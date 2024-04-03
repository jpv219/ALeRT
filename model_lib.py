##########################################################################
#### Regression model library
#### Author: Juan Pablo Valdes
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
from scikeras.wrappers import KerasRegressor

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
    
class MLPRegressorWrapper(MLP):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_model(self):
        
        # wrap Keras net as a sklearn regressor object
        net = self.build_net()

        return KerasRegressor(model = net,verbose=1)
    
class MLPWrapper(MLP):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self):
        return self.build_net()
    
############################# MODEL CONFIG ###################################

class ModelConfig:
        
    ## Regressor instances, labels and hyperparameters
    model_names = {'dt': 'Decision_Tree', 
                    'xgb': 'XGBoost', 
                    'rf': 'Random_Forest',
                    'svm': 'Support_Vector_Machine',
                    'knn': 'K_Nearest_Neighbours',
                    'mlp_reg': 'MLP_Wrapped_Regressor',
                    'mlp': 'Multi_Layer_Perceptron'}
    
    wrapper_dict = {
        'dt': DecisionTreeWrapper, 
        'xgb': XGBoostWrapper, 
        'rf': RandomForestWrapper,
        'svm': SVMWrapper,
        'knn': KNNWrapper,
        'mlp_reg': MLPRegressorWrapper,
        'mlp': MLPWrapper
    }

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
        'mlp_reg': {'n_dense' : 2,
                    'n_shallow': 2,
                    'n_nodes_d': 128,
                    'n_nodes_s': 64,
                    'n_epochs' : 100,
                    'batch_size' : 1,
                    'act_fn': 'relu',
                    'lr': 0.01},
        'mlp': {'n_dense' : 2,
                'n_shallow': 2,
                'n_nodes_d': 128,
                'n_nodes_s': 64,
                'n_epochs' : 100,
                'batch_size' : 1,
                'act_fn': 'relu',
                'lr': 0.01}
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