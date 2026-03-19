import sys
import pickle
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats
import scikit_posthocs as sp
from paths import DATA_DIR


def print_values_LOSO(file_path, metric="test_accuracy"):
    """
    return average and std of LOSO classification for all models and all metrics
    """
    with open(file_path, 'rb') as fp:
        results=pickle.load(fp)

    models_accuracy=[]
    for result_m in results:

        model=list(result_m.keys())[0]
        # pdb.set_trace()
        models_accuracy.append(result_m[model][metric])
        print(model)
        for s in list(result_m[model].keys()):

            print('\t\t\t\t%s\t%.3f +- %3f (std err +- %3f)' % (s, np.mean(result_m[model][s]), np.std(result_m[model][s]), stats.sem(result_m[model][s])))
    print(stats.friedmanchisquare(*models_accuracy))
    # combine three groups into one array
    data = np.array(models_accuracy)

    # perform Nemenyi post-hoc test
    print(sp.posthoc_nemenyi_friedman(data.T))


def create_barplot_performance_signals(model="XGBOOST", figure_path=[], metric="test_accuracy", paths=[]):
    """
    idx model=which model we want to plot
    """
    performances=[]
    for idx, path in enumerate(paths):
        performances.append(return_values(path, model, metric))

    average=np.mean(performances, 1)
    #std = np.std(performances, 1)
    sterr=stats.sem(performances, axis=1)
    print(average)

    print(stats.friedmanchisquare(*performances))
    # combine three groups into one array
    data = np.array(performances)

    # perform Nemenyi post-hoc test
    print(sp.posthoc_nemenyi_friedman(data.T))


    signals=["ALL", "EEG", "SCH", "SCF"]
    x_pos = np.arange(len(signals))
    plt.figure()
    fig, ax = plt.subplots()
    ax.bar(x_pos, average, yerr=sterr, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(signals)
    ax.set_title('Accuracy of Classifier')
    ax.yaxis.grid(True)

    plt.savefig(figure_path+".png")
    plt.savefig(figure_path + ".pdf")
    plt.show()
    

def return_values(file_path, model, metric):
    """
    reutrn values of a single model. not LOSO but files with training and test
    """
    with open(file_path, 'rb') as fp:
        results=pickle.load(fp)
    result = [ (list(results[ i ].keys())[ 0 ] , i) for i in range(0 , 7) ]
    result_dict = {key : value for key , value in result}
    results=results[result_dict[model]]
    print(results)
    performance=results[list(results.keys())[0]][metric]
    return performance


def plot_models(file_path, figure_path, metric="test_accuracy"):

    """
    plot for all models the average and std of LOSO classification for metric
    """

    with open(file_path, 'rb') as fp:
        results=pickle.load(fp)
    avperf=[]
    #stdperf=[]
    stderrperf = []
    for result_m in results:

        model=list(result_m.keys())[0]
        print(model)
        avperf.append(np.mean(result_m[model][metric]))
        #stdperf.append(np.std(result_m[model][metric]))
        stderrperf.append(stats.sem(result_m[model][metric]))

    models_name=[list(result_m.keys())[0] for result_m in results]
    x_pos = np.arange(len(models_name))
    plt.figure()
    fig, ax = plt.subplots()
    ax.bar(x_pos, avperf, yerr=stderrperf, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(metric)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_name,fontsize=8 )
    ax.set_title(metric)
    ax.yaxis.grid(True)
    plt.savefig(figure_path+metric+".png")
    plt.savefig(figure_path + metric+".pdf")
    plt.show()


def print_performance(file_path, figure_path, print_values_=True, plot_models_=True,metric="test_accuracy"):
    """
    main function. use flag to decide which functions to use
    """
    if print_values_:
        print_values_LOSO(file_path, metric=metric)

    if plot_models_:
        plot_models(file_path, figure_path, metric=metric)

# paths=[os.path.join(RESULTS_DIR_CLASS , path) for path in ["best_params_features_automatic", "best_params_features_automatic_EEG", "best_params_features_automatic_SCH", "best_params_features_automatic_SCF"]]
# create_barplot_performance_signals(idx_model=1, figure_path=RESULTS_DIR_CLASS+"comparison", metric="test_accuracy", paths=paths)