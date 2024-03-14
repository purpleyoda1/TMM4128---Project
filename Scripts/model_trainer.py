"""
This script creates and trains a ML model. Step by step it does the following:
 - Set the algorithm you want to use to True
 - Define a pipeline and what parameters to search trough
 - Create and train a model with the scikit GridSearchCV
 - Save the results to a pdf
 - Save the model, labels, and scaler as joblibs in 'Datasets/Music_genre/Models/KNN'

 How to use:
 - Select algorithm by changing the booleans in the top
 - Change pipeland and param_grid if needed
 - Run
"""


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import os
import joblib
import utilities
import pandas as pd


use_KNN = True
use_RandomForest = False


if use_KNN:
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])

    param_grid = {
            'knn__n_neighbors': range(2, 22),
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan']
        }
    #Filepaths
    model_name = "KNN"
    results_filename = 'OG_params'
    pdf_filepath = 'Results/KNN/' + results_filename + '.pdf'
    txt_filename = 'Results/KNN/' + results_filename + '.txt'
    model_path = 'Models/KNN/knn.joblib'
    scaler_path = 'Models/KNN/knn_scaler.joblib'

if use_RandomForest:
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state= 42))
        ])
    
    param_grid = {
        'rf__n_estimators' : [100, 200, 300],
        'rf__max_depth' : [None, 10, 20, 30],
        'rf__min_samples_split' : [2, 5, 10],
        'rf__min_samples_leaf' : [1, 2, 4]
    }

    #Filepaths
    model_name = "RF"
    results_filename = 'first_try'
    pdf_filepath = 'Results/RF/' + results_filename + '.pdf'
    txt_filename = 'Results/RF/' + results_filename + '.txt'
    model_path = 'Models/RF/rf.joblib'
    scaler_path = 'Models/RF/rf_scaler.joblib'


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
    utilities.save_data_frame_as_pdf(results, pdf_filepath)
    scaler = grid_search.best_estimator_.named_steps['scaler']

    #Use grid search to train a model
    model = grid_search.best_estimator_
    model.fit(X_train, y_train)
    
    #Test the model
    utilities.test_model(txt_filename, model_name= model_name, path=False, model=model, params= grid_search.best_params_)

    #Save the model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return 0


if __name__ == "__main__":
    create_train_and_save_model_scaled()




















""" def create_knn_model(n_neighbors = 3):
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

    return acc, conf_matrix, class_report """