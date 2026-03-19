import json
import pickle
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
import feature_selection
import grid_search_models
import loso_classify
import read_results
import shap_helpers
import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)
import helpers
import warnings
import numpy as np
from paths import INIT_DIR, RESULTS_DIR_CLASS, TRIALS_PATH, SUBJECTS_PATH, cohorts, cohort
warnings.filterwarnings('ignore')


if __name__ == "__main__":  # important for multithreading
    filename = os.path.join(INIT_DIR, "Classification", "args_class.json")
    with open(filename) as f:
        args = json.load(f)
    ## LOAD data
    TrialsFeatures = helpers.read_dataset(TRIALS_PATH)
    TrialsFeatures = helpers.select_cohort(TrialsFeatures, **cohorts, dataset_path=SUBJECTS_PATH)
    TrialsFeatures = helpers.select_single_area(TrialsFeatures, area=args['gridsearch']['area'])
    # feature selection
    if args['flags']['flag_feat_selection']:
        # if flag_feat_selection True, perform feature select. Methods: methods supported by pandas corr
        feature_selection.compute_feature_selection_crosscorr(data=TrialsFeatures,
                                                                corr_thresh=args['feat_selection']['corr_thresh'],
                                                                method=args['feat_selection']['method'],
                                                                dropfeat_path=RESULTS_DIR_CLASS + args['feat_selection'][
                                                                    'dropfeat_path'] + ".pkl")
    with open(RESULTS_DIR_CLASS + args['feat_selection']['dropfeat_path'] + ".pkl", 'rb') as fp:
        features_corr_to_drop = pickle.load(fp)
    print(features_corr_to_drop)

    ## GRIDSEARCH
    X=TrialsFeatures.copy()
    y=X['pain']
    groups = X["id"]
    features_not_physio = ['Area', 'B', 'id', 'iTrial', 'NRS', 'pain']
    X = X.drop(features_corr_to_drop + features_not_physio, axis=1)
    X = helpers.drop_by_pattern(X , ["cohort"])

    #if we want to use only one signal select here
    X=helpers.select_feature_signals(X, EEG=args['signal']['EEG'], SCH=args['signal']['SCH'], SCF=args['signal']['SCF'])
    if cohort=="chronic":
        GS_size=25
    elif cohort=="healthy":
        GS_size=10

    X_train=X[groups.isin(groups.unique()[0:GS_size])]
    y_train=y[groups.isin(groups.unique()[0:GS_size])]
    groups_train =groups[groups.isin(groups.unique()[0:GS_size])]
    X_test=X[groups.isin(groups.unique()[GS_size:])]
    y_test=y[groups.isin(groups.unique()[GS_size:])]
    groups_test =groups[groups.isin(groups.unique()[GS_size:])]
    if args['flags']['flag_gridsearch']:
        print("Selected Features:")
        print(X.columns)
        print("Starting grid search..... \n")
        grid_search_models.compute_grid_search_each_model(X_train, y_train, gridsearchdir=RESULTS_DIR_CLASS + args['gridsearch'][
            'gridsearch_path'] + ".pkl", scoring="accuracy")

    ## LOAD MODELS
    with open(RESULTS_DIR_CLASS + args['gridsearch']['gridsearch_path'] + ".pkl", 'rb') as fp:
        print('\n ****** GRIDSEARCH RESULTS****** \n')
        models = pickle.load(fp)
        print(models)

    ## MAIN CLASSIFICATION LOSO
    if args['flags']['flag_LOSO']:
        print('\n ****** LOSO CLASSIFICATION ****** \n')
        scoring = {'accuracy': make_scorer(accuracy_score),
                    'precision': make_scorer(precision_score, zero_division=0),
                    'recall': make_scorer(recall_score),
                    'f1_score': make_scorer(f1_score),
                    }

        scores = loso_classify.subject_leave_one_group_out(X_test, groups_test, y_test, models, scoring)
        with open(RESULTS_DIR_CLASS + args['LOSO']['loso_path']+'.pkl', 'wb') as fp:
            pickle.dump(scores, fp)

    read_results.print_performance(RESULTS_DIR_CLASS + args['LOSO']['loso_path']+'.pkl', figure_path=RESULTS_DIR_CLASS + args['LOSO']['loso_path'], print_values_=True, plot_models_=True, metric=args['LOSO']['metric'])

    if args[ 'flags' ][ 'flag_ROCAUC' ] :
    # to compute ROC AUC for a single model over loso classification
        loso_classify.subject_k_fold_single(X_test, groups_test, y_test, model=models[ "XGBOOST" ])

    ## explainability
    shap_flag_XGB = args['XAI']['shap_flag_XGB']
    if shap_flag_XGB:
        X_shap,shap_values, shap_values_obj = shap_helpers.shap_GroupKFold(models[ "XGBOOST" ],X_test, y_test, groups_test,  clf_name="XGBOOST")
        predicted_class = 1 ##PAIN
        plt.figure(figsize=(20,30))
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, plot_size=(10,10), max_display=48)
        plt.savefig(RESULTS_DIR_CLASS+args['LOSO']['loso_path']+'XGB_bar.pdf')
        plt.figure(figsize=(20,30))
        shap.summary_plot(shap_values, X_shap, plot_type="violin", show=False, plot_size=(5,5), max_display=15)
        plt.savefig(RESULTS_DIR_CLASS+args['LOSO']['loso_path']+'XGB_violin.pdf')
        plt.figure(figsize=(20,30))
        #shap.plots.heatmap(shap_values , max_display=12, show=False)
        plt.show()
        col_ascending=X_shap.columns[np.argsort(np.abs(shap_values).mean(0))]


        #save in results dir and args XAI selectfeat_path

        with open(RESULTS_DIR_CLASS + args['XAI']['selectfeat_path'] + ".pkl", 'wb') as fp:
            pickle.dump(col_ascending[-args['XAI']['selectfeat_num']:], fp)

    if args['XAI']['feature_importance_flag_XGB']:
        # visualize the average feature importance using a bar chart
        avg_importance=shap_helpers.feature_importance_XGB(X_test, y_test, groups_test, model=models[ "XGBOOST" ])
        sorted_indices = np.argsort(avg_importance)[ : :-1 ]

        # get the names of the features based on the original dataset columns
        feature_names = X.columns[ sorted_indices ]
        features_to_plot=20
        # visualize the average feature importance using a bar chart
        plt.figure(figsize=(20 , 30))
        plt.bar(range(features_to_plot) , avg_importance[ sorted_indices ][:features_to_plot])
        plt.xticks(range(features_to_plot) , feature_names[:features_to_plot] , rotation=90)
        plt.ylabel('Feature Importance')
        plt.savefig(RESULTS_DIR_CLASS + args[ 'LOSO' ][ 'loso_path' ] + 'XGB_feature_importance.pdf')
        plt.show()


    shap_flag_RF=args['XAI']['shap_flag_RF']

    if shap_flag_RF:
        X_shap, shap_values = shap_helpers.shap_GroupKFold(models["Random Forest"],X_test, y_test, groups_test, )
        # importance plot
        predicted_class = 1 ##PAIN
        plt.figure(figsize=(20,30))
        shap.summary_plot(shap_values[predicted_class], X_shap, plot_type="bar", show=False, plot_size=(10,10))
        plt.savefig(RESULTS_DIR_CLASS+args['LOSO']['loso_path']+'_RF_bar.pdf')
        plt.figure(figsize=(20,30))
        shap.summary_plot(shap_values[predicted_class], X_shap, plot_type="violin", show=False, plot_size=(5,5))
        plt.savefig(RESULTS_DIR_CLASS+args['LOSO']['loso_path']+'_RF_violin.pdf')
        plt.show()

