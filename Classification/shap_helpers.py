import numpy as np
import shap
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, permutation_test_score
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, make_scorer


def shap_GroupKFold(clf , data , labels , groups , explainerClass=shap.TreeExplainer , clf_name=None) :
    """
    Compute shap values for a classifier using GroupKFold cross validation.
    clf: classifier
    data: data to be used for training
    labels: labels to be used for training
    groups: groups to be used for training
    explainerClass: shap explainer class
    clf_name: name of the classifier

    returns: X_test, shap_values

    """
    shap.initjs()
    groupKFold = GroupKFold(n_splits=len(set(groups)))
    list_shap_values = list()
    list_shap_values_obj = list()
    list_test_sets = list()
    for train_index , test_index in groupKFold.split(data , labels , groups=groups) :
        X_train , X_test = data.iloc[ train_index ] , data.iloc[ test_index ]
        y_train , y_test = labels.iloc[ train_index ] , labels.iloc[ test_index ]

        # training model
        clfNew = clone(clf)
        clfNew.fit(X_train , y_train)

        # explaining model
        if explainerClass is shap.LinearExplainer :
            masker = shap.maskers.Independent(data=X_train)  # or X_test?
            explainer = explainerClass(clfNew , masker)
            shap_values = explainer.shap_values(X_test)
        else :
            explainer = explainerClass(clfNew)
            shap_values_obj = explainer(X_test)
            shap_values = explainer.shap_values(X_test)

        # for each iteration we save the test_set index and the shap_values
        list_shap_values.append(shap_values)
        list_shap_values_obj.append(shap_values_obj)
        list_test_sets.append(test_index)

    # combining results from all iterations
    test_set = list_test_sets[ 0 ]
    shap_values = np.array(list_shap_values[ 0 ])
    for i in range(1 , len(list_test_sets)) :
        test_set = np.concatenate((test_set , list_test_sets[ i ]) , axis=0)
        if explainerClass is shap.LinearExplainer :
            shap_values = np.concatenate((shap_values , list_shap_values[ i ]) , axis=0)
        elif clf_name == "XGBOOST" :
            shap_values = np.concatenate((shap_values , list_shap_values[ i ]) , axis=0)
        else :
            shap_values = np.concatenate((shap_values , np.array(list_shap_values[ i ])) , axis=1)
    X_test = data.iloc[ test_set ]

    return X_test , shap_values , list_shap_values_obj

def feature_importance_XGB (data, labels, groups,model):
    """
    Compute feature importance for a classifier using GroupKFold cross validation.
    clf: classifier
    data: data to be used for training
    labels: labels to be used for training
    groups: groups to be used for training
    """
    feature_importances = [ ]
    # iterate over the folds of the LOSO CV
    groupKFold = GroupKFold(n_splits=len(set(groups)))
    for train_index , test_index in groupKFold.split(data , labels , groups=groups) :
        X_train , X_test = data.iloc[ train_index ] , data.iloc[ test_index ]
        y_train , y_test = labels.iloc[ train_index ] , labels.iloc[ test_index ]
        model.fit(X_train , y_train)
        # get the feature importances for the trained model
        importance = model.feature_importances_
        #importance=model.get_booster().get_score(importance_type=type)
        # append the feature importances to the list
        feature_importances.append(importance)

    # compute the average feature importance across all folds
    avg_importance = np.mean(feature_importances , axis=0)
    return avg_importance
