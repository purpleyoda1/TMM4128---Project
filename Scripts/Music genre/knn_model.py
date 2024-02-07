"""
This script creates and trains a KNN model. Step by step it does the following:
 -Define a pipeline and what parameters to search trough
 -Create and train a model with the scikit GridSearchCV
 -Save the results to a pdf
 -Save the model, labels, and scaler as joblibs in 'Datasets/Music_genre/Models/KNN'

 How to use:
 By changing the pipeline and param_grid in the top of the script you will change what the gridsearch changes and tries
 when its making the model
 Change pdf_name to save a new pdf instead of writing over the last when changing the search
"""


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import os
import joblib
import data_loader as dl
import utilities
import pandas as pd



pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

param_grid = {
        'knn__n_neighbors': range(2, 12),
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    }

pdf_name = 'knn_results_bal_split.pdf'


def create_knn_model(n_neighbors = 3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def make_predictions(model, X_test):
    prediction = model.predict(X_test)
    return prediction

def evalution(y_test, prediction):
    acc = accuracy_score(y_test, prediction)
    conf_matrix = confusion_matrix(y_test, prediction)
    class_report = classification_report(y_test, prediction)

    return acc, conf_matrix, class_report

def create_and_train_gridsearch(X_train, y_train):

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    return grid_search

def create_train_and_save_model_scaled():
    #prepare data
    X_train, X_test, y_train, y_test = utilities.get_data()

    #Do a grid search
    grid_search = create_and_train_gridsearch(X_train, y_train)
    print(f"Best parameters are: {grid_search.best_params_}")
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values(by='rank_test_score')
    results= results[['rank_test_score', 'mean_test_score', 'params']]
    utilities.save_data_frame_as_pdf(results, 'Results/KNN/' + pdf_name)
    scaler = grid_search.best_estimator_.named_steps['scaler']

    #Use grid search to train a model
    model = grid_search.best_estimator_
    model.fit(X_train, y_train)
    
    #Test the model
    utilities.test_model('KNN_bal_split', path=False, model=model)

    #Save the model, genre labels, scaler, and scaled training data
    joblib.dump(model, 'Models/KNN/knn.joblib')
    joblib.dump(scaler, 'Models/KNN/knn_scaler.joblib')

    return 0


if __name__ == "__main__":
    create_train_and_save_model_scaled()