from tcn import TCN, tcn_full_summary
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from plot_metric.functions import BinaryClassification
import warnings
warnings.filterwarnings("ignore")

# preprocess library
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score

def Evaluation(y_test, y_prob, y_label, labels):
    bc =  BinaryClassification(y_test, y_prob, labels=labels)

    # plots
    plt.figure(figsize=(15,10))
    plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    bc.plot_roc_curve()
    plt.subplot2grid((2,6), (0,3), colspan=2)
    bc.plot_precision_recall_curve()
    plt.show()

    def F(beta, precision, recall):
    
        """
        Function that calculate f1, f2, and f0.5 scores.
        
        @params: beta, Float, type of f score
                precision: Float, average precision
                recall: Float, average recall
        
        @return: Float, f scores
        """
        
        return (beta*beta + 1)*precision*recall / (beta*beta*precision + recall)
    
    # precision, recall, and f1 f2 scores
    precision, recall, _ = precision_recall_curve(y_test, y_label)
    fpr, tpr, _ = roc_curve(y_test, y_label)

    print('f1 score {0:.4f}:'.format(F(1, np.mean(precision), np.mean(recall))))
    print('f2 score {0:.4f}:'.format(F(2, np.mean(precision), np.mean(recall))))
    print('precision {0:.4f}:'.format(precision_score(y_test, y_label)))
    print('recall {0:.4f}:'.format(recall_score(y_test, y_label)))
    print('AUPRC {0:.4f}:'.format(auc(recall, precision)))
    print('AUROC {0:.4f}:'.format(auc(fpr, tpr)))
    print('Acc {0:.4f}:'.format(accuracy_score(y_test, y_label)))

    # report
    bc.print_report()


def sumarized_report(model_list, y_test):

    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

    for clf in model_list:
        fpr, tpr, _ = roc_curve(y_test, eval(f'y_prob_{clf}'))
        auc = roc_auc_score(y_test, eval(f'y_prob_{clf}'))

        precision, recall, thresholds = precision_recall_curve(y_test, eval(f'y_prob_{clf}'))
        avg_prc = average_precision_score(y_test, eval(f'y_prob_{clf}'))

        result_table = result_table.append({'classifiers':clf,
                                            'True': y_test,
                                            'Predicted': eval(f'y_prob_{clf}'),
                                            'fpr':fpr, 
                                            'tpr':tpr, 
                                            'auc':auc,
                                            'precision': precision,
                                            'recall': recall,
                                            'pr_auc': avg_prc,
                                            'threshold': thresholds}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    return result_table



def multi_ROC(result_table):
    fig = plt.figure(figsize=(8,6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'], 
                result_table.loc[i]['tpr'], 
                label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
        
    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.show()



def multi_PRC(result_table):
    fig = plt.figure(figsize=(8,6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['recall'], 
                result_table.loc[i]['precision'], 
                label="{}, PRC={:.3f}".format(i, result_table.loc[i]['pr_auc']))
        
    #plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision", fontsize=15)

    plt.title('PR Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.show()