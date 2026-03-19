from sklearn.model_selection import GridSearchCV , RepeatedStratifiedKFold , RandomizedSearchCV, GroupShuffleSplit
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier , AdaBoostClassifier , BaggingClassifier
import xgboost as xgb

random_seed = 0

models = {
    'Decision Tree' : DecisionTreeClassifier() ,
    'Bagging' : BaggingClassifier() ,
    'LDA' : LinearDiscriminantAnalysis() ,
    'Logistic Regression' : LogisticRegression() ,
    'Extremely Randomized Trees' : ExtraTreesClassifier() , 'Ada Boost' : AdaBoostClassifier() ,
    'Random Forest' : RandomForestClassifier() ,
          'XGBOOST' : xgb.XGBClassifier()}

parameters_all = {
    'Decision Tree' : {'max_depth' : [ 2 , 3 , 5 , 10 , 20 ] , 'min_samples_leaf' : [ 5 , 10 , 20 , 50 , 100 ] ,
                       'criterion' : [ "gini" , "entropy" ] , 'random_state' : [ random_seed ]} ,
    'Bagging' : {"n_estimators" : [ 10 , 20 , 50 , 80 , 90 , 100 , 200 , None ] , 'bootstrap' : [ False , True ] ,
                 'max_features' : [ 0.5 , 0.7 , 1.0 ] , 'max_samples' : [ 0.5 , 0.7 , 1.0 ] ,
                 'random_state' : [ random_seed ]} ,

    'LDA' : {'solver' : [ "svd" , "lsqr" ] , } ,
    'Logistic Regression' : {"C" : np.logspace(-3 , 3 , 7) , "penalty" : [ "l1" , "l2" ] ,
                             "solver" : [ "lbfgs" , "liblinear" ] , 'random_state' : [ random_seed ]} ,
    'Extremely Randomized Trees' : {'bootstrap' : [ True , False ] ,
                                    'max_depth' : [ 6 , 10 , 20 , 50 , 80 , 90 , 100 , 200 , None ] ,
                                    'max_features' : [ 'log2' , 'sqrt' ] , 'min_samples_leaf' : [ 1 , 2 , 4 ] ,
                                    'min_samples_split' : [ 2 , 5 , 10 ] ,
                                    'n_estimators' : [ 10 , 50 , 100 , 200 , 400 , 600 , 800 , 1000 , 1200 , 1400 ,
                                                       1600 , 1800 , 2000 ] , 'random_state' : [ random_seed ] ,

                                    } ,
    'Ada Boost' : {"n_estimators" : [ 10 , 50 , 100 , 200 , 500 , 1000 ] , 'random_state' : [ random_seed ]

                   } ,
    'Random Forest' : {'bootstrap' : [ True , False ] ,
                                          'max_depth' : [ 6 , 10 , 20 , 50 , 80 , 90 , 100 , 200 , None ] ,
                                          'max_features' : [ 'log2' , 'sqrt' ] , 'min_samples_leaf' : [ 1 , 2 , 4 ] ,
                                          'min_samples_split' : [ 2 , 5 , 10 ] ,
                                          'n_estimators' : [ 10 , 50 , 100 , 200 , 400 , 600 , 800 , 1000 , 1200 ,
                                                             1400 , 1600 , 1800 , 2000 ] ,
                                          'random_state' : [ random_seed ]} ,

    'XGBOOST' : {'min_child_weight' : [ 1 , 5 , 10 ] , 'gamma' : [0, 0.1, 0.5 , 1 , 1.5 , 2 , 5 ] ,
                 'subsample' : [ 0.6 , 0.8 , 1.0 ] , 'colsample_bytree' : [ 0.6 , 0.8 , 1.0 ] ,
                 'max_depth' : [ 3 , 4 , 5 , 6 , 10 ] , 'random_state' : [ random_seed ] ,
                 'n_estimators' : [ 10 , 50 , 100 , 200 , 500 , 1000 ], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1], "reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100], "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100]}}


def compute_grid_search_each_model(X , y , gridsearchdir="./gridsearch.pkl", scoring="accuracy") :
    """
    Compute grid search for each model
    :param X: feature set
    :param y: label set
    :param gridsearchdir: path to save best models
    :return:

    """
    best_models = {}
    for model_name in sorted(models) :
        print(model_name)
        clf = models[ model_name ]
        params = parameters_all[ model_name ]
        best_model = randomized_grid_search_algorithm(clf , params , X , y , scoring=scoring)
        best_models[ model_name ] = best_model
    with open(gridsearchdir , 'wb') as fp :
        pickle.dump(best_models , fp)
    return


def randomized_grid_search_algorithm(classifier , parameters , X , y , n_splits=5 , n_repeats=3 , scoring="accuracy") :
    """
    Randomized grid search algorithm
    :param classifier: classifier
    :param parameters: parameters
    :param X: feature set   (n_samples, n_features)
    :param y: label set     (n_samples, )
    :param n_splits: number of splits   (default 5)
    :param n_repeats: number of repeats (default 3)
    :return: best model (classifier)
    """
    param_comb = 5
    cv = RepeatedStratifiedKFold(n_splits=n_splits , n_repeats=n_repeats , random_state=42)
    random_search = RandomizedSearchCV(classifier , param_distributions=parameters , n_iter=param_comb ,
                                       scoring=scoring , n_jobs=-1 , cv=cv.split(X , y) , verbose=1 , random_state=42 ,
                                       refit=True)

    gs = random_search.fit(X , y)
    # summarize the results of your GRIDSEARCH

    print("Best score: %f using %s" % (gs.best_score_ , gs.best_params_))
    means = gs.cv_results_[ 'mean_test_score' ]
    best_model = gs.best_estimator_
    return best_model


