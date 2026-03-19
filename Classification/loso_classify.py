import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV, LeaveOneGroupOut, cross_validate, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay, auc,roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
import helpers
import multiprocessing

import xgboost as xgb
from paths import DATA_DIR, RESULTS_DIR_CLASS


random_seed = 0
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, zero_division=0),  # zero_division to avoid warnings in notebook
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score)
           }

scoring_multilabel = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, zero_division=0, average='macro'),  # zero_division to avoid warnings in notebook
           'recall': make_scorer(recall_score, zero_division=0, average='macro'),
           'f1_score': make_scorer(f1_score, average='macro'),
           }

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=random_seed),
    'Bagging': BaggingClassifier(DecisionTreeClassifier(),random_state=random_seed),
    'LDA': LinearDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(solver='liblinear',random_state=random_seed),
    'Extremely Randomized Trees': ExtraTreesClassifier(random_state=random_seed),
    'Ada Boost': AdaBoostClassifier(DecisionTreeClassifier(), random_state=random_seed),
    'Random Forest': RandomForestClassifier(random_state=random_seed),
    'XGBOOST': xgb.XGBClassifier()
}


def cross_validation_algorithm(data, labels, models=models, standardized=False, scorers=None, multilabel=False):
    """
    cross validation of models with Kfold. return scores
    """
    scores = {}
    if multilabel:
        cv=KFold(n_splits=5)
        if scorers == None:
            scorers = scoring_multilabel
    else:
        cv=StratifiedKFold(n_splits=5)
        if scorers == None:
            scorers = scoring
    for model_name in sorted(models):
        #clf = OneVsRestClassifier(models[model_name]) needed if classifier does not support directly multi class
        clf = models[model_name]
        ## NEW NOEMI
        if standardized:
            clf_p=make_pipeline(StandardScaler() ,clf)
        else:
            clf_p=clone(clf)

        score = cross_validate(clf_p, data, labels, cv=cv.split(data, labels), scoring=scorers)
        scores[model_name] = score
    helpers.print_scores_cv(scores)
    return scores


def subject_k_fold_algorithm(data, groups, labels, models, scoring):
    """
    LOSO cross validation dividing datasets depending on groups. return scores
    """
    groupKFold = GroupKFold()
    scores = {}
    for model_name in sorted(models):
        clf = models[model_name]
        score = cross_validate(clf, data, labels, cv=groupKFold,
                               groups=groups, scoring=scoring)
        scores[model_name] = score
    helpers.print_scores_cv(scores)
    return scores


def subject_k_fold_single(data, groups, labels, model):
    """
    LOSO cross validation dividing datasets depending on groups. return scores
    """
    # Run classifier with cross-validation and plot ROC curves
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    cv = LeaveOneGroupOut().split(data, labels, groups=groups)
    for i, (train, test) in enumerate(LeaveOneGroupOut().split(data, labels, groups=groups)):
        model.fit(data.iloc[train], labels.iloc[train])
        viz = RocCurveDisplay.from_estimator(
            model,
            data.iloc[test],
            labels.iloc[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example",
    )
    ax.legend(loc="lower right")
    plt.savefig(RESULTS_DIR_CLASS+"./ROCAUC.pdf")
    plt.show()
    

def subject_leave_one_group_out(data, groups, labels, models, scoring, standardized=False):
    """
    multiprocessing on Models with LOSO classification.
    official function to get metrics for each LOSO subject test
    """
    params_list = [list(models.values()), list(models.keys()), data, labels, groups, scoring, standardized]
    results = customMultiprocessing(multiprocessing_model_fit, params_list, pool_size=8)
    #print(results)
    return results

def multiprocessing_model_fit(clf, model_name, data, labels, groups, scoring, standardized=False):
    """
    function as input of the costum multiprocessing. fit and save score of crossval
    """
    #clf = models[model_name]
    print(model_name)
    score={}
    ## NEW NOEMI
    if standardized:
        clf_p = make_pipeline(StandardScaler() , clf)
    else:
        clf_p=clone(clf)

    score[model_name] = cross_validate(clf_p, data, labels, cv=LeaveOneGroupOut().split(data, labels, groups=groups),
                           scoring=scoring, verbose=1)
    print(f" fine {model_name}")
    return score


def customMultiprocessing(f, params_list, pool_size=8):
    '''
    Execute a function f using multiprocessing \n
    :param f: function to exectute
    :param params_list: list of list of params. Element inside must be of the same lenght (number of iterations) or of lenght 1. In case of lenght 1 they will be repeated. MUST be a list of lists!
    :param pool_size: number of processes to use
    :return: list of element containing the return of each execution of the function
    '''
    print("Beginning multiprocessing")
    lengths = [len(el) if type(el) == list else 1 for el in params_list]
    base_el = np.max(lengths)
    for lens in lengths:
        if (lens == base_el) or (lens == 1):
            continue
        else:
            raise Exception(
                "Params list must have elements of the same lengths or of length 1, found: {}".format(lengths))
    final_params = [el if type(el) == list else [el]*base_el for el in params_list]
    print("Parameters ok")
    pool = multiprocessing.Pool(pool_size)
    print("Pool done")
    return pool.starmap(f, zip(*final_params))


def train_test_algorithm(data, labels, models, scoring):
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.25, random_state=1)
    scores = fit(models, X_train, X_test, y_train, y_test)
    #print_scores(scores)
    return scores


def fit(models, X_train, X_test, y_train, y_test, figure_path=RESULTS_DIR_CLASS):
    scores = {}
    for model_name in sorted(models):
        clf = models[model_name]
        clf.fit(X_train, y_train)
        scores[model_name] = helpers.score(clf, X_test, y_test, scoring)
        y_pred=clf.predict(X_test)
        #### keep probabilities for pain
        y_pred_proba=clf.predict_proba(X_test)[:,1]

        plt.figure()
        cf_matrix = confusion_matrix(y_test,y_pred, normalize="true")
        ax = sns.heatmap(cf_matrix,annot=True,cmap='Blues')
        ax.set_title('Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels([ 'pain','no pain' ])
        ax.yaxis.set_ticklabels([ 'pain','no pain' ])
        ## Display the visualization of the Confusion Matrix.
        plt.savefig(figure_path+model_name + "_cm.png")
        plt.savefig(figure_path+model_name+ "_cm.pdf")


        fpr,tpr,_ = roc_curve(y_test,y_pred_proba)
        auc = roc_auc_score(y_test,y_pred_proba)

        plt.figure()
        # create ROC curve
        plt.plot(fpr,tpr,label="AUC=" + str(auc))
        plt.axline([ 0,0 ],[ 1,1 ], linestyle='--',label='No Skill')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend()
        plt.savefig(figure_path+model_name + "_roc.png")
        plt.savefig(figure_path+model_name+ "_roc.pdf")
        plt.show()
    return scores


def columns_to_drop(data):
    features = ["iqr", "range", "mad", "std", "ZC", "cohort"]
    trial_correlated = [col for col in data.columns if any(
        feat in col for feat in features)]
    trial_correlated2 = ['Area', 'B', 'id', 'iTrial', "NRS"]
    columns_to_drop = trial_correlated+trial_correlated2+["BL"]

    data.drop(columns_to_drop, inplace=True, axis=1)
    return data


def signals_to_drop(data, signals=[]):
    columns_to_drop = [col for col in data.columns if any(
        feat in col for feat in signals)]
    data.drop(columns_to_drop, inplace=True, axis=1)
    return data

