"""
This script is for splitting the music genre dataset into a training set and a test set

When running it, it will take in the original csv file, split it into training and test gy genre to assure even distribution 
of datapoints in each genre in the sets, and then save them in their own csv file. When needed, the **get_data()** function from 
utilities should be used to fetch the data. 

This ensures that the training and testing sets doesnt change while using them for different models, and will therefore help 
ensure valid and reliable results

The data_split functions also returns the divided sets, but this should not be needed. Use get_data instead. 
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

def data_loader(filepath):
    data = pd.read_csv(filepath)

    X = data.drop(["filename", "length", "label"], axis=1)
    y = data["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    joblib.dump(le.classes_, 'Models/genre_labels.joblib')

    return X, y_encoded

def data_split(X, y, test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify = y)

    X_train.to_csv('Datasets/Training_test_splits/X_train.csv', index=False)
    X_test.to_csv('Datasets/Training_test_splits/X_test.csv', index=False)
        #Save y as np array
    np.savetxt('Datasets/Training_test_splits/y_train.csv', y_train, delimiter=',')
    np.savetxt('Datasets/Training_test_splits/y_test.csv', y_test, delimiter=',')


    """ pd.DataFrame(y_train, columns= ['label']).to_csv('Datasets/Training_test_splits/y_train.csv', index=False)
    pd.DataFrame(y_test, columns= ['label']).to_csv('Datasets/Training_test_splits/y_test.csv', index=False) """

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X, y_encoded = data_loader('Datasets/features_30_sec.csv')
    data_split(X, y_encoded)












#Gammel splitter med manuell oppdeling av sjanger, hvil i fred all tid brukt
""" def data_split(X, y, test_size=0.2, random_state=42):
    n = 100
    X_chunks = [X[i:i +n] for i in range(0, X.shape[0], n)]
    y_chunks = [y[i:i +n] for i in range(0, y.shape[0], n)]

    X_train, X_test, y_train, y_test = [], [], [], []

    for X_chunk, y_chunk in zip(X_chunks, y_chunks):
        divided_X_train, divided_X_test, divided_y_train, divided_y_test = train_test_split(X_chunk, y_chunk, test_size=test_size, random_state=random_state)
        X_train.append(divided_X_train)
        X_test.append(divided_X_test)
        y_train.append(divided_y_train)
        y_test.append(divided_y_test)

    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)

    #Save X as np array
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    np.savetxt('Datasets/Music_genre/Training_test_splits/X_train.csv', X_train, delimiter=',')
    np.savetxt('Datasets/Music_genre/Training_test_splits/X_test.csv', X_test, delimiter=',')

    #Save X as pandas DataFrame
    X_train = pd.concat(X_train)
    X_test = pd.concat(X_test)
    X_train.to_csv('Datasets/Music_genre/Training_test_splits/X_train.csv', label= False, index=False)
    X_test.to_csv('Datasets/Music_genre/Training_test_splits/X_test.csv', index=False)

    #Save y as np array
    np.savetxt('Datasets/Music_genre/Training_test_splits/y_train.csv', y_train, delimiter=',')
    np.savetxt('Datasets/Music_genre/Training_test_splits/y_test.csv', y_test, delimiter=',')

    #Save y as pandas DataFrame
    pd.DataFrame(y_train, header= None).to_csv('Datasets/Music_genre/Training_test_splits/y_train.csv', index=False)
    pd.DataFrame(y_test, header = None).to_csv('Datasets/Music_genre/Training_test_splits/y_test.csv', index=False)


    return X_train, X_test, y_train, y_test """