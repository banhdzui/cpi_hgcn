'''
Created on 27 Apr 2020

@author: danhbuithi
'''
import numpy as np 
import matplotlib.pyplot as plt


from lifelines.utils.concordance import concordance_index
from sklearn.metrics._classification import f1_score
from sklearn.metrics._ranking import roc_auc_score, roc_curve
from sklearn.metrics._regression import mean_squared_error


def evaluate_classification_result(y_trues, y_preds, y_pred_scores, draw_curve=False):
    print('F1 score each class', f1_score(y_trues, y_preds, average=None))
    print('F1 score ', f1_score(y_trues, y_preds))
    print('AUC score ', roc_auc_score(y_trues, y_pred_scores))
    if draw_curve == True:
        fpr_rf, tpr_rf, _ = roc_curve(y_trues, y_pred_scores)
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_rf, tpr_rf, label='RT + LR')
        plt.show()
        
    
def evaluate_regression_result(y_trues, y_pred_scores):
    print('MSE: ', mean_squared_error(y_trues, y_pred_scores))
    print('CI: ', concordance_index(y_trues, y_pred_scores))
        
def compute_importance(scores, threshold = 0):
    
    x = np.copy(scores)
    x[x < threshold] = 0
    
    b = np.sum(x)
    x = x/b 
    return x 
    
def draw_heatmap(sequence, scores, length_limit=30, max_length = None, output=None):
        
    if max_length is None:
        max_length = scores.shape[0]
    #max_length = 800
    nrows = int(max_length/length_limit)

    if max_length % length_limit > 0: nrows += 1
    padded_scores = np.pad(scores[:max_length], (0, length_limit*nrows-max_length), 'constant', constant_values=0)
    heatmap_values = np.reshape(padded_scores, (nrows,-1))
    #heatmap_values[heatmap_values < 0.1] = 0
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap_values, cmap='YlOrRd')
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('[feature importance]', rotation=-90, va="bottom")
    
    for index, c in enumerate(sequence):
        i = int(index/length_limit)
        j = index % length_limit
        ax.text(j, i, c+'.',ha="center", va="center", color='black', fontsize=7)
    fig.tight_layout()
    
    if output is None:
        plt.show()
    else:
        plt.savefig(output, figsize=[250, 250],dpi=300)