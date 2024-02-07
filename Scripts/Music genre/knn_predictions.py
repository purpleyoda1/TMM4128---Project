import joblib
import data_loader as dl
from sklearn.metrics import accuracy_score
import extract_from_audio as audio
from sklearn.preprocessing import StandardScaler
import os


def knn_predicitons(filepath):
    #Load data
    X, y_encoded, class_names = dl.data_loader(filepath)

    #Load model
    knn_model = joblib.load('Models/knn.joblib')
    knn_predicitons = knn_model.predict(X)
    acc = accuracy_score(y_encoded, knn_predicitons)

    print(f"The following predictions was made: \n{knn_predicitons}\n\nIt got an accuracy of {acc}")
    
    return knn_predicitons, acc

def knn_predicitons_audiofile(filepath):

    #Load audio file
    X = audio.extract_from_audio(filepath)

    #Load model, label
    knn_model = joblib.load('Models/knn.joblib')
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
    knn_predicitons_audiofile('Datasets/Music_genre/External_tests/Bad_Guy.mp3')
    knn_predicitons_audiofile('Datasets/Music_genre/External_tests/Born_To_Run.mp3')
    knn_predicitons_audiofile('Datasets/Music_genre/External_tests/The_Devil_In_I.mp3')
    knn_predicitons_audiofile('Datasets/Music_genre/External_tests/We_Made_You.mp3')
    knn_predicitons_audiofile('Datasets/Music_genre/External_tests/Toxic.mp3')
    knn_predicitons_audiofile('Datasets/Music_genre/External_tests/Crossroad.mp3')
    knn_predicitons_audiofile('Datasets/Music_genre/External_tests/Dont_Stop_Me_Now.mp3')
    knn_predicitons_audiofile('Datasets/Music_genre/External_tests/Master_Of_Puppets.mp3')
    knn_predicitons_audiofile('Datasets/Music_genre/External_tests/Juicy.mp3')
    knn_predicitons_audiofile('Datasets/Music_genre/External_tests/Hurt.mp3')

if __name__ == "__main__":
    knn_test_external()