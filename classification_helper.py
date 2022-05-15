import os
import numpy as np
import pandas as pd
import seaborn as sns
from math import exp
from matplotlib import pyplot as plt
from models import SVM, LinearRegressionModel, LogisticRegressionModel, Perceptron


def classify(model, a, x):

    if isinstance(model, LinearRegressionModel) or isinstance(model, Perceptron) :
        return sum(a_i * x[i] for i, a_i in enumerate(a))
    
    elif isinstance(model, LogisticRegressionModel):
        exp_a_transpose_x = exp(a.T.dot(x))
        return exp_a_transpose_x / (1 + exp_a_transpose_x) 
    
    else:
        raise NotImplementedError("Only Linear, Logistic regression, Perceptron and SVM \
            models are implemented so far.")


def confusion_matrix(model, A, b, solution):
    matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    
    classification_threshold = 0.5 if isinstance(model, LogisticRegressionModel) else 0
    
    for i, label in enumerate(b):
        if isinstance(model, SVM):
            classification = 1 if label * A[i].T.dot(solution) < 1 else -1
        else:
            classification = classify(model, A[i], solution)
        
        if classification > classification_threshold:
            if isinstance(model, LinearRegressionModel):
                if classification * label > classification_threshold: matrix['TP'] += 1
                else: matrix['FP'] += 1
            else:
                if label == 1: matrix['TP'] += 1
                else: matrix['FP'] += 1
        else:
            if isinstance(model, LinearRegressionModel):
                if classification * label < classification_threshold: matrix['TN'] += 1 
                else: matrix['FN'] += 1
            else:
                if label == 1: matrix['FN'] += 1
                else: matrix['TN'] += 1

    return matrix


def confusion_matrix_plot(m, title, filename):
    mat = np.array([[m['TP'], m['FP']], [m['TN'], m['FN']]])
    classes = ['Positive', 'Negative']
    mat_pd = pd.DataFrame(mat, index=classes, columns=['True', 'False'])

    f = plt.figure(dpi=120)    
    sns.heatmap(mat_pd, annot=True, cbar=False, fmt='g', cmap='coolwarm')

    accuracy = (m['TP'] + m['TN']) / np.sum(mat)
    plt.suptitle(f'Confusion matrix ({title})\n' + r'Accuracy = $\dfrac{TP + TN}{Total} = $' + str(accuracy))
    plt.xlabel('True Class')
    plt.ylabel('Predicted Class')
    os.makedirs(r'figs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(fr'figs\{filename}.png')
    

    
def line_func(coeffs):
    def line(x):
        return (coeffs[0] + coeffs[1] * x) / -coeffs[2]
    return line