from sklearn.metrics import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train_test_reports(best_model, x_test, x_train, y_test, y_train):
    y_score = best_model.predict_proba(x_test)[:,1]
    pred_labels = best_model.predict(x_test)
    fits = best_model.predict_proba(x_train)[:,1]
    fit_labels = best_model.predict(x_train)

    print("--------Training Data----------")
    print(classification_report(y_true = y_train, y_pred = fit_labels))
    print("AUROC:", roc_auc_score(y_train, fits))
    print("--------Testing Data----------")
    print(classification_report(y_true = y_test, y_pred = pred_labels))
    print("AUROC:", roc_auc_score(y_test, y_score))
    
    
def plot_roc_curve(best_model, x_test, x_train, y_test, y_train):
    
    y_score = best_model.predict_proba(x_test)[:,1]
    y_true = y_test
   
    fig, ax = plt.subplots(1, 2, figsize = (12, 5))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    ax[0].plot(fpr, tpr, linewidth = 2)
    ax[0].plot([0,1], [0,1], "k--")
    ax[0].set_xlabel('False Positive Rate', fontsize = 18)
    ax[0].set_ylabel('True Positive Rate', fontsize = 18)

    ax[1].plot(rec, prec, linewidth = 2)
    ax[1].set_xlabel("Recall", fontsize = 18)
    ax[1].set_ylabel("Precision", fontsize = 18)
    plt.show()
    
def print_confusion_matrix(best_model, x_test, x_train, y_test, y_train):
    y_score = best_model.predict(x_test)
    
    print ('Confusion Matrix :') 
    print(confusion_matrix(y_test, y_score)) 
    
    