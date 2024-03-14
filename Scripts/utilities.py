import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import joblib
import extract_from_audio
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import seaborn as sns
import re


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
    filepath_data = 'Datasets/features_30_sec.csv'
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
    X_train = pd.read_csv('Datasets/Training_test_splits/X_train.csv')  
    X_test = pd.read_csv('Datasets/Training_test_splits/X_test.csv')
    y_train = np.loadtxt('Datasets/Training_test_splits/y_train.csv')
    y_test = np.loadtxt('Datasets/Training_test_splits/y_test.csv')

    return X_train, X_test, y_train, y_test



def create_cm_heatmap(cm, model_name):
    plt.figure(figsize=(20, 20))
    genre_labels = joblib.load('Models/genre_labels.joblib')
    sns.heatmap(cm, annot= True, fmt= 'g', cmap= 'Reds', cbar= False, xticklabels= genre_labels, yticklabels= genre_labels)
    plt.xlabel("Predictet labels", 15)
    plt.ylabel("True labels", 15)
    plt.title("Confusion matrix" + model_name)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Results/' + model_name +' Vizualisations')

    plt.show()



def test_model(results_dir, model_name, path= True, model_path= None, model= None, params= None):
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

    #Print the predictions
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    #Make and save confusion matrix vizualisation
    create_cm_heatmap(conf_matrix, model_name)

    #Save the values
    with open(f'{results_dir}', "w") as f:
        f.write(f'Parameters used: {params}\n\n')
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(f"Confusion matrix:\n")
        f.write(f"{conf_matrix}\n\n")
        f.write(f"Classification report: \n")
        f.write(f"{class_report}")



def retrieve_test_results(filepath):
    params, accuracy, conf_matrix, class_report = None, None, None, None

    with open(filepath, 'r') as f:
        results = f.readlines()

    for line in results:
        if line.startswith("Parameters used:"):
            params = line.strip().split(": ")[1]
        elif line.startswith("Accuracy"):    
            accuracy = line.strip().split(": ")[1]
        elif line.startswith("Confusion matrix"):    
            matrix = []
            for m_line in results[results.index(line) + 1:]:
                if m_line.strip() == "":
                    break
                matrix.append((list(map(int, re.findall(r'\d+', m_line)))))
            conf_matrix = np.array(matrix)
        elif line.startswith('Classification report:'):
            report_lines = results[results(line)+1:]
            class_report = "".join(report_lines)
            break
    
    return params, accuracy, conf_matrix, class_report



