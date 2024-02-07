import pandas as pd
import matplotlib.pyplot as plt 
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import joblib
import extract_from_audio
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os


def save_data_frame_as_pdf(df, filepath):

    data = [df.columns.to_list()] + df.to_numpy().tolist()

    # Create a PDF object and a Table object
    pdf = SimpleDocTemplate(filepath, pagesize=letter)
    table = Table(data)

    # Add style to the Table
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), (0.7, 0.7, 0.7)),
        ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), (0.95, 0.95, 0.95)),
    ])
    table.setStyle(style)

    # Build the PDF
    elements = [table]
    pdf.build(elements)
    


#Unfinished, dont know if its needed
def compare_dataset_and_input(genre = 'blues', number = '00001'):
    #Get the dataset csv line
    filepath_data = 'Datasets/Music_genre/features_30_sec.csv'
    genre_index = genre_labels.transform(genre)[0]
    skiprows = genre_index*100 + number  
    filename = genre + '/' + 'genre' +'.' + number + '.wav'
    
    genre_labels = joblib.load('Models/knn_genre_labels.joblib')
    print(f"Number of rows to skip: {skiprows}")
    data_values = pd.read_csv(filepath_data, skiprows = skiprows, nrows = 1)
    data_values = data_values.drop([filename, 661794, genre], axis=1)

    #Get the audio file values
    filepath_audio = 'Datasets/Music_genre/genres_original/' + filename
    data_audio = extract_from_audio.extract_from_audio(filepath_audio)



def get_data():
    X_train = pd.read_csv('Datasets/Music_genre/Training_test_splits/X_train.csv')  
    X_test = pd.read_csv('Datasets/Music_genre/Training_test_splits/X_test.csv')
    y_train = pd.read_csv('Datasets/Music_genre/Training_test_splits/y_train.csv')
    y_test = pd.read_csv('Datasets/Music_genre/Training_test_splits/y_test.csv')

    return X_train, X_test, y_train, y_test



def test_model(model_name, path= True, model_path= None, model= None):
    _,X_test,_, y_test = get_data()

    #If it needs to load a model from a file
    if (path):
        model = joblib.load(model_path)

    #Make predictions
    predictions = model.predict(X_test)

    #Test the predictions
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    results = {
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
        'Classification report': class_report
    } 

    #Print and save predictions
    print(results)
    results_df = pd.DataFrame(results)
    results_path = 'Results/KNN/' + model_name + '.csv'
    results_df.to_csv(results_path)

    
