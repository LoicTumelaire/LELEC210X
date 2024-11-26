import os
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from classification.datasets import Dataset
from classification.utils.audio_student import AudioUtil, Feature_vector_DS
from classification.utils.plots import (
    plot_decision_boundaries,
    plot_specgram,
    show_confusion_matrix,
)
from classification.utils.utils import accuracy
import time

#function to perform grid search
def perform_grid_search(model, param_grid, kf, X, y, model_name, save_fold=False):
    start_time = time.time()
    print("\n-----------------------------------------------")
    print("Start grid search for", model_name, "model...\n")
    
    score = make_scorer(lambda y_true, y_false: accuracy(y_true, y_false), greater_is_better=True)

    grid = GridSearchCV(model, param_grid, cv=kf, scoring=score, n_jobs=-1, verbose=1, return_train_score=True)
    grid.fit(X, np.ravel(y))

    if save_fold:
        save_folds(model_name, grid)

    stop_time = time.time()
    print(f'Finished : execution time for {model_name} model: {stop_time-start_time:.2f} seconds')
    
    return grid

def save_folds(model_name, grid_search):
    with open(f"results/{model_name}_grid_search.csv", "w") as f:
        results = pd.DataFrame(grid_search.cv_results_)
        results = results[results.columns.drop(list(results.filter(regex='split')))]
        results = results.drop(columns=['mean_fit_time', 'mean_score_time', 'std_score_time']) #time stats
        results.drop(columns=['params'], inplace=True)
        results.sort_values(by='rank_test_score', inplace=True)
        results['mean_test_score'] = -results['mean_test_score']
        results['mean_train_score'] = -results['mean_train_score']
        results.to_csv(f, index=False, lineterminator='\n')

def eval_visualise(model, model_name, X_train, y_train, X_test, y_test, grid=True):
    # Compute RMSE
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    accuracy_train = accuracy(y_train_pred, y_train)*100
    accuracy_test = accuracy(y_test_pred, y_test)*100

    print(f"Result for {model_name} model:")
    print(f"\tAccuracy on training set: {accuracy_train:.3f}%")
    print(f"\tAccuracy on test set: {accuracy_test:.3f}%")

    if grid:
        print(f"\nBest hyperparameters for {model_name} model:")
        print(model.best_params_)

    save_final_score = False
    if save_final_score:
        f.close()
        sys.stdout = sys.__stdout__
        f = open(f"results/{model_name}_results.txt", "r")
        print(f.read())
        f.close()

    show_confusion_matrix(y_test_pred, y_test, classnames)

fm_dir = "data/feature_matrices/"  # where to save the features matrices
model_dir = "data/models/"  # where to save the models

dataset = Dataset()
classnames = dataset.list_classes()

myds = Feature_vector_DS(dataset, Nft=512, nmel=20, duration=950, shift_pct=0.2)

"Random split of 70:30 between training and validation"
train_pct = 0.7

featveclen = len(myds["fire", 0])  # number of items in a feature vector
nitems = len(myds)  # number of sounds in the dataset
naudio = dataset.naudio  # number of audio files in each class
nclass = dataset.nclass  # number of classes
nlearn = round(naudio * train_pct)  # number of sounds among naudio for training

data_aug_factor = 1 #no data augmentation => maybe add later
class_ids_aug = np.repeat(classnames, naudio * data_aug_factor)

"Compute the matrixed dataset, this takes some seconds, but you can then reload it by commenting this loop and decommenting the np.load below"
"""
X = np.zeros((data_aug_factor * nclass * naudio, featveclen))
for s in range(data_aug_factor):
    for class_idx, classname in enumerate(classnames):
        for idx in range(naudio):
            featvec = myds[classname, idx]
            X[s * nclass * naudio + class_idx * naudio + idx, :] = featvec
np.save(fm_dir + "feature_matrix_2D_svm.npy", X)
"""
X = np.load(fm_dir+"feature_matrix_2D_svm.npy")

"Labels"
y = class_ids_aug.copy()

print(f"Shape of the feature matrix : {X.shape}")
print(f"Number of labels : {len(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True) # Normalization
X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True) # Normalization

#PCA
pca = PCA(n_components=14)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#grid search for different models

# Split the dataset into training and validation subsets
n_splits = 10
kf = StratifiedKFold(n_splits=n_splits, shuffle=True) #normaly K-fold is used to determine the best hyperparameters => strange to use it here
accuracies_knn = np.zeros((n_splits,))

# SVM
svm = SVC()
param_grid = {
    "C": [20,100, 1000],
    "gamma": [0.1, 0.01, 0.001, 0.0001],
    "kernel": ["rbf", "linear", "poly", "sigmoid"],
}
grid_svm = perform_grid_search(svm, param_grid, kf, X_train_pca, y_train, "SVM", save_fold=False)
eval_visualise(grid_svm, "SVM", X_train_pca, y_train, X_test_pca, y_test)

# KNN
#redo pca
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn = KNeighborsClassifier()
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
}
grid_knn = perform_grid_search(knn, param_grid, kf, X_train_pca, y_train, "KNN", save_fold=False)
eval_visualise(grid_knn, "KNN", X_train_pca, y_train, X_test_pca, y_test)

# Gaussian process
#no grid search for gaussian process
from sklearn.gaussian_process import GaussianProcessClassifier
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

gp = GaussianProcessClassifier()
gp.fit(X_train_pca, y_train)
eval_visualise(gp, "Gaussian Process", X_train_pca, y_train, X_test_pca, y_test, grid=False)
