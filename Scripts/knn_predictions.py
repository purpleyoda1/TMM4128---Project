import joblib
from sklearn.metrics import accuracy_score
import extract_from_audio as audio
from sklearn.preprocessing import StandardScaler
import os
import utilities


def knn_predicitons(filepath):
    #Load data
    _, X_test, _, y_test = utilities.data_loader(filepath)

    #Load model
    knn_model = joblib.load('Models/KNN/knn.joblib')
    knn_predicitons = knn_model.predict(X_test)
    acc = accuracy_score(y_test, knn_predicitons)

    print(f"The following predictions was made: \n{knn_predicitons}\n\nIt got an accuracy of {acc}")
    
    return knn_predicitons, acc

def knn_predicitons_audiofile(filepath):

    #Load audio file
    X = audio.extract_from_audio(filepath)

    #Load model, label
    knn_model = joblib.load('Models/KNN/knn.joblib')
    knn_label = joblib.load('Models/genre_labels.joblib')

    #Make prediction
    knn_predictions = knn_model.predict(X)

    #Get song name and print result
    song_name = os.path.basename(filepath)
    song_name = os.path.splitext(song_name)[0]
    genre = knn_label[knn_predictions[0]]
    print(f"{song_name} is a {genre} song")

    return knn_predicitons




def knn_test_external():
    knn_predicitons_audiofile('Datasets/External_tests/Bad_Guy.mp3')
    knn_predicitons_audiofile('Datasets/External_tests/Born_To_Run.mp3')
    knn_predicitons_audiofile('Datasets/External_tests/The_Devil_In_I.mp3')
    knn_predicitons_audiofile('Datasets/External_tests/We_Made_You.mp3')
    knn_predicitons_audiofile('Datasets/External_tests/Toxic.mp3')
    knn_predicitons_audiofile('Datasets/External_tests/Crossroad.mp3')
    knn_predicitons_audiofile('Datasets/External_tests/Dont_Stop_Me_Now.mp3')
    knn_predicitons_audiofile('Datasets/External_tests/Master_Of_Puppets.mp3')
    knn_predicitons_audiofile('Datasets/External_tests/Juicy.mp3')
    knn_predicitons_audiofile('Datasets/External_tests/Hurt.mp3')

if __name__ == "__main__":
    knn_test_external()