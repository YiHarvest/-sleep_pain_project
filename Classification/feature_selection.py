import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pdb
import os, sys
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
import helpers
from paths import TRIALS_PATH, RESULTS_DIR_CLASS

value=25
plt.rc('font', size=value)          # controls default text sizes
plt.rc('axes', titlesize=value)     # fontsize of the axes title
plt.rc('axes', labelsize=value)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=value)    # fontsize of the tick labels
plt.rc('ytick', labelsize=value)    # fontsize of the tick labels
plt.rc('legend', fontsize=value)

"""
following functions derived by https://towardsdatascience.com/are-you-dropping-too-many-correlated-features-d1c96654abe6
"""

def corrX_new(df, cut=0.8, method="pearson"):
    """
    This function takes a dataframe df and a correlation cut-off value cut and returns a list of variables to drop.

    Parameters
    ----------
    df : pandas dataframe
    cut : float correlation cut-off value
    method : string correlation method, can be any of the following: {‘pearson’, ‘kendall’, ‘spearman’}

    Returns
    -------
    list : list of variables to drop

    """
    # Get correlation matrix and upper triagle
    corr_mtx = df.corr(method=method).abs()
    avg_corr = corr_mtx.mean(axis=1)
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool))

    dropcols = list()

    res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target',
                                 'v2.target', 'corr', 'drop']))

    for row in range(len(up) - 1):
        col_idx = row + 1
        for col in range(col_idx, len(up)):
            if (corr_mtx.iloc[row, col] >= cut):
                if (avg_corr.iloc[row] > avg_corr.iloc[col]):
                    dropcols.append(row)
                    drop = corr_mtx.columns[row]
                else:
                    dropcols.append(col)
                    drop = corr_mtx.columns[col]

                s = pd.Series([corr_mtx.index[row],
                               up.columns[col],
                               avg_corr[row],
                               avg_corr[col],
                               up.iloc[row, col],
                               drop],
                              index=res.columns)

                res = res.append(s, ignore_index=True)

    dropcols_names = calcDrop(res)

    return (dropcols_names)

def calcDrop(res):
    # All variables with correlation > cutoff
    all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))

    # All unique variables in drop column
    poss_drop = list(set(res['drop'].tolist()))

    # Keep any variable not in drop column
    keep = list(set(all_corr_vars).difference(set(poss_drop)))

    # Drop any variables in same row as a keep variable
    p = res[res['v1'].isin(keep) | res['v2'].isin(keep)][['v1', 'v2']]
    q = list(set(p['v1'].tolist() + p['v2'].tolist()))
    drop = (list(set(q).difference(set(keep))))

    # Remove drop variables from possible drop
    poss_drop = list(set(poss_drop).difference(set(drop)))

    # subset res dataframe to include possible drop pairs
    m = res[res['v1'].isin(poss_drop) | res['v2'].isin(poss_drop)][['v1', 'v2', 'drop']]

    # remove rows that are decided (drop), take set and add to drops
    more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
    for item in more_drop:
        drop.append(item)

    return drop

def select_by_corr(data, cut=0.8, method="pearson", dropfeat_path=RESULTS_DIR_CLASS + "features.pkl"):
    corr = data.corr(method=method) # default numeric only
    plot_corr(corr, annot=False, plot_path=dropfeat_path)
    col_to_drop=corrX_new(data, cut=cut, method=method)
    #save features to be dropped
    with open(dropfeat_path, 'wb') as fp:
        pickle.dump(col_to_drop, fp)
    print("{} features have been dropped:".format(len(col_to_drop)))
    print(col_to_drop, "\n")
    return

def compute_feature_selection_crosscorr(data=None, corr_thresh=0.8, method="pearson", dropfeat_path=RESULTS_DIR_CLASS + "features.pkl"):
    """
    Compute feature selection using cross-correlation

    Parameters
    ----------
    data : pandas dataframe
    corr_thresh : float correlation cut-off value
    method : string correlation method, can be any of the following: {‘pearson’, ‘kendall’, ‘spearman’}
    dropfeat_path : string path to save features to be dropped

    Returns
    -------
    list : list of variables to drop
    """
    if data is None:
        raise Exception('A dataset must be passed')
    print("CORRELATION  between physiological features - Starting ... ")
    #exclude object features
    data = data.select_dtypes(exclude='O')
    select_by_corr(data, cut=corr_thresh, method=method, dropfeat_path=dropfeat_path)
    plt.show()
    return

def plot_corr(corr, threshold=0, corr_method="pearson", annot=False, plot_path=RESULTS_DIR_CLASS):
    """
    Plot a graphical correlation matrix for a dataframe.
    Input:
        corr: pandas DataFrame
        threshold: float between 0 and 1, default 0. Only correlations higher than this value are shown.
        corr_method: string, default "pearson"
            Method used to calculate the correlation.
            Can be any of the following: {‘pearson’, ‘kendall’, ‘spearman’}
        annot: boolean, default False
            If True, write the data value in each cell.
        plot_path: string, default "./"
            Path to save the plot
    """

    mask = (corr>=threshold) | (corr <=-threshold)
    plt.figure(figsize=(60,50))
    heatmap = sns.heatmap(corr[mask], vmin=-1, vmax=1, annot=annot, cmap="seismic")
    heatmap.set_title('Correlation - ' + corr_method, fontdict={'fontsize':12}, pad=12)
    plt.savefig(plot_path+"corr_{}.pdf".format(corr_method))
    plt.show()
    return corr


# option: correlation with label
def drop_uncorrelated(data, corr, threshold, label):
    return data.loc[:, abs(corr[label]) >= threshold]


def plot_corr_with_label(corr, label):
    sorted_corr_with_label = corr[label][corr[label].abs().sort_values().keys()]
    sorted_corr_with_label.drop(label, inplace=True)
    plt.figure(figsize=(60,50))
    plt.barh(sorted_corr_with_label.keys(), sorted_corr_with_label.values)
    plt.show()
    return sorted_corr_with_label


if __name__ == '__main__':
    TrialsFeatures=helpers.read_dataset(TRIALS_PATH)
    compute_feature_selection_crosscorr(data=TrialsFeatures, corr_thresh=0.9, method="pearson")

