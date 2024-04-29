"""
This script creates and trains a ML model. Step by step it does the following:
 - Set the algorithm you want to use to True
 - Define a pipeline and what parameters to search trough
 - Create and train a model with the scikit GridSearchCV
 - Save the results to a pdf
 - Save the model, labels, and scaler as joblibs in 'Datasets/Music_genre/Models/KNN'

 How to use:
 - Select algorithm by changing the model_name in the top
 - Change pipeland and param_grid if needed
 - Run
"""


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import utilities
import pandas as pd
import feature_selection as fs


#Set model_name to either KNN, RF, or SVM depending on what model you want to train
model_name = "RF"
feature_selection = True
#Modify pipeline and param_grid as you wish


if model_name == "KNN":
    pipeline = Pipeline([
            #('scaler', StandardScaler()),
            #('knn', KNeighborsClassifier())
        ])

    param_grid = {
            'knn__n_neighbors': range(1, 21),
            #'knn__weights': ['uniform', 'distance'],
            #'knn__metric': ['euclidean', 'manhattan']
        }
    #Filepaths
    results_filename = 'simple'
    pdf_filepath = 'Results/BestParameters/KNN/' + results_filename + '.pdf'
    txt_filename = 'Results/BestParameters/KNN/' + results_filename + '.txt'
    model_path = 'Models/KNN/KNN.joblib'
    scaler_path = 'Models/KNN/KNN_scaler.joblib'
elif model_name == "RF":
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state= 42))
        ])
    
    param_grid = {
        'rf__n_estimators' : [100, 200, 300],
        'rf__max_depth' : [20, 30, None],
        'rf__min_samples_split' : [2, 5], 
        'rf__min_samples_leaf' : [1, 2, 4],
        'rf__criterion': ['gini', 'entropy']
    }

    #Filepaths
    results_filename = 'everything'
    pdf_filepath = 'Results/BestParameters/RF/' + results_filename + '.pdf'
    txt_filename = 'Results/BestParameters/RF/' + results_filename + '.txt'
    model_path = 'Models/RF/RF.joblib'
    scaler_path = 'Models/RF/RF_scaler.joblib'
elif model_name == "SVM":
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC())
        ])

    param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            #'svm__kernel': ['linear'],
            'svm__gamma': ['scale', 'auto'] 
        }
    # Filepaths
    results_filename = 'everything'
    pdf_filepath = 'Results/BestParameters/SVM/' + results_filename + '.pdf'
    txt_filename = 'Results/BestParameters/SVM/' + results_filename + '.txt'
    model_path = 'Models/SVM/SVM.joblib'
    scaler_path = 'Models/SVM/SVM_scaler.joblib'




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
    #utilities.save_data_frame_as_pdf(results, pdf_filepath)
    scaler = grid_search.best_estimator_.named_steps['scaler']

    #Use grid search to train a model
    model = grid_search.best_estimator_
    
    #Test the model
    utilities.test_model(txt_filename, model_name= model_name, path=False, model= model, params= grid_search.best_params_, create_heatmap= True)

    #Save the model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return 0

def train_model_w_feature_selection(method, enable_print= False):
    #This function is alot more computationally expensive than it needs to be 
    #but in the final weeks we only need something that works, not something optimized
    X_train, X_test, y_train, y_test = utilities.get_data()
    train_accuracies = []
    test_accuracies = []
    feature_counts = []
    best_test_acc = 0
    best_model = None
    best_grid_search = None
    best_num_features = None


    for i in range(57, 1, -1):
        feature_counts.append(i)
        #Reduce the number of features iteratively
        if method == "univariate":
            _, _, X_train, X_test = fs.univariate_feature_selection(num_features= i)

        elif method == "recursive":
            _, X_train, X_test = fs.recursive_feature_elimnination(num_features= i)
        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of X_test: {X_test.shape}")


        #Perform gridsearch on reduced training set
        grid_search = create_and_train_gridsearch(X_train, y_train)
        model = grid_search.best_estimator_

        #Get accuracies
        train_predicitions = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predicitions)
        test_predicitions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predicitions)
        acc_differance = abs(train_accuracy - test_accuracy)

        #Log accuracies
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if enable_print:
            print(f"Num of features: {i}")
            print(f"Training acc: {train_accuracy}")
            print(f"Test acc: {test_accuracy}\n\n")

        #Arbitrarily chosen limits to avoid overfitting
        if train_accuracy < 0.95: #and acc_differance < 0.2:
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                best_model = model
                best_grid_search = grid_search
                best_num_features = i


    #Now that the best model is found, we test, save, and plot
    print(f"Training acc: {train_accuracy}")
    print(f"Test acc: {test_accuracy}")
    print(f"Num features: {best_num_features}")

    utilities.test_model('Results/BestParameters/' + model_name + '/FS_' + results_filename, model_name= model_name, path=False, model= best_model, params= best_grid_search.best_params_, create_heatmap= True, num_features=best_num_features)
    
    utilities.plot_train_test_accuracies(train_accuracies, test_accuracies, feature_counts, 'Results/FeatureSelection/' + model_name + '_' + method + '_' + results_filename)



if __name__ == "__main__":
    if feature_selection:
        train_model_w_feature_selection("univariate")
    else:
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