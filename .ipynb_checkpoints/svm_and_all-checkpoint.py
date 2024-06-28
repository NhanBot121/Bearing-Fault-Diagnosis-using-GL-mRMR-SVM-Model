import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,  plot_confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import os

def preprocess_data(data, test_size=180, random_state=2):
    # Slip to train and test
    train_data, test_data = train_test_split(data, test_size=test_size, stratify=data['label'], random_state=random_state)

    # Standardize the data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data.iloc[:, 1:])
    test_data_scaled = scaler.transform(test_data.iloc[:, 1:])

    return train_data, test_data, train_data_scaled, test_data_scaled

def train_svc(train_data_scaled, train_labels):
    model = SVC()
    model.fit(train_data_scaled, train_labels)
    return model


# Model Evaluation
def evaluate_svm(model, train_data_scaled, test_data_scaled, train_labels, test_labels):
    ## Get the predictions from the model and
    ## create a confusion matrix and a classification report

    train_predictions = model.predict(train_data_scaled)
    test_predictions = model.predict(test_data_scaled)

    ## Confusion Matrices
    train_confu_matrix = confusion_matrix(train_labels, train_predictions)
    test_confu_matrix = confusion_matrix(test_labels, test_predictions)

    class_report = classification_report(y_pred=test_predictions, y_true=test_labels, output_dict=True)
    return train_confu_matrix, test_confu_matrix, class_report


def tune_svm(train_data_scaled, train_labels, param_grid):
    grid_search = GridSearchCV(SVC(), param_grid, n_jobs=-1, cv=10)
    grid_search.fit(train_data_scaled, train_labels)
    return grid_search.best_estimator_, grid_search.best_params_


## Plot
def plot_confusion_matrices(model, train_data_scaled, test_data_scaled, train_data, test_data, show=True):
    plt.figure(figsize=(18, 8))

    plt.subplot(121)
    plot_confusion_matrix(model, train_data_scaled, train_data['label'], display_labels=np.sort(train_data['label'].unique()),
                          cmap=plt.cm.Blues, ax=plt.gca())
    plt.title('Training Confusion Matrix')

    plt.subplot(122)
    plot_confusion_matrix(model, test_data_scaled, test_data['label'], display_labels=np.sort(test_data['label'].unique()),
                          cmap=plt.cm.Blues, ax=plt.gca())
    plt.title('Test Confusion Matrix')
    
    if show==True:
        plt.show()
        
param_grid = {
    "C": [1, 10, 45, 47, 49, 50, 51, 55, 100, 128, 300, 500],
    "gamma": [0.01, 0.05, 0.1, 0.2, 0.5, 1, 5],
    "kernel": ["rbf"]
}

def svm_model(path='./12k_DE_0', file='selected_12_fts.csv', random_state=21, param_grid=param_grid):
    
    data_file = f'./{path}/{file}'
    data = pd.read_csv(data_file)
    train_data, test_data, train_data_scaled, test_data_scaled = preprocess_data(data, test_size=180, random_state=random_state)
    best_model, best_params = tune_svm(train_data_scaled, train_data['label'], param_grid)
    
    return best_model, best_params


def classify_svm(path='./12k_DE_0', file='selected_12_fts.csv', model=SVC(), test_size=180, random_state=21, show=True):
    data_file = f'./{path}/{file}'
    data = pd.read_csv(data_file)
    train_data, test_data, train_data_scaled, test_data_scaled = preprocess_data(data, test_size=test_size, random_state=random_state)


    train_conf_matrix, test_conf_matrix, class_report = evaluate_svm(model,
                                                                    train_data_scaled,
                                                                    test_data_scaled,
                                                                    train_data['label'],
                                                                    test_data['label'])
    #plot_confusion_matrices(model, train_data_scaled, test_data_scaled, train_data, test_data, show=show)
    
    return class_report, test_conf_matrix




