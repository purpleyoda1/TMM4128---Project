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
import re
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import feature_selection as fs



def decode_labels(y):
    y = pd.DataFrame(y, columns= ['label'])
    genre_labels = joblib.load('Models/genre_labels.joblib')
    label_dict = {index: label for index, label in enumerate (genre_labels)}
    y['label'] = y['label'].map(label_dict)

    return y

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



def get_data(scale= False, get_scaled= False):
    X_train = pd.read_csv('Datasets/Training_test_splits/X_train.csv')  
    X_test = pd.read_csv('Datasets/Training_test_splits/X_test.csv')
    y_train = np.loadtxt('Datasets/Training_test_splits/y_train.csv')
    y_test = np.loadtxt('Datasets/Training_test_splits/y_test.csv')

    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    if get_scaled:
        X_train = pd.read_csv('Datasets/Training_test_splits/Scaled/X_train_scaled.csv')  
        X_test = pd.read_csv('Datasets/Training_test_splits/Scaled/X_test_scaled.csv')
 
    return X_train, X_test, y_train, y_test



def create_cm_heatmap(cm, model_name):
    plt.figure(figsize=(8, 8))
    genre_labels = joblib.load('Models/genre_labels.joblib')
    sns.heatmap(cm, annot= True, fmt= 'g', cmap= 'Reds', cbar= False, xticklabels= genre_labels, yticklabels= genre_labels)
    plt.xlabel("Predicted labels", fontsize= 15)
    plt.ylabel("True labels", fontsize= 15)
    plt.title("Confusion matrix for " + model_name)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Results/ConfusionMatrixes/confusion_matrix' + model_name)
    plt.show()



def test_model(results_dir, model_name, path= True, model_path= None, model= None, params= None, create_heatmap= False, num_features= 57, method= "univariate"):
    X_train, X_test, y_train, y_test = get_data()

    cf_name = model_name
    if num_features != 57:
        if method == "univariate":
            _, _, X_train, X_test = fs.univariate_feature_selection(num_features= num_features)
            cf_name += '_' + method

        elif method == "recursive":
            _, X_train, X_test = fs.recursive_feature_elimnination(num_features= num_features)
            cf_name += '_' + method


    #If it needs to load a model from a file
    if (path):
        model = joblib.load(model_path)

    #Check training asccuracy
    train_predicitions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predicitions)

    #Make predictions on test set
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    #Print the predictions
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    #Make and save confusion matrix vizualisation
    if create_heatmap:
        create_cm_heatmap(conf_matrix, cf_name)

    #Save the values
    with open(f'{results_dir}', "w") as f:
        f.write(f'Parameters used: {params}\n\n')
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(f"Confusion matrix:\n")
        f.write(f"{conf_matrix}\n\n")
        f.write(f"Classification report: \n")
        f.write(f"{class_report}")
        f.write(f"Training accuracy: {train_accuracy}")



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

def plot_model_parameter_range(model_name, param_name, param_range, x_ticks= None):
    #For plotting the accuracy of a model over a range of parameters
    model_path = 'Models/' + model_name + '/' + model_name + '.joblib'
    model = joblib.load(model_path)

    training_acc = []
    testing_acc = []
    #Get data
    X_train, X_test, y_train, y_test = get_data(scale= True)

    if model_name == "KNN":
        for value in param_range:
            #Make a new model
            new_model = KNeighborsClassifier(n_neighbors= value)

            #Train model
            new_model.fit(X_train, y_train)

            #Find training accuracy
            train_pred = new_model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            training_acc.append(train_acc)

            #Find test accuracy
            test_pred = new_model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            testing_acc.append(test_acc)

    plt.figure(figsize= (6, 5))
    plt.plot(param_range, training_acc, label= 'Training accuracy', lw= 2, color= 'black')
    plt.plot(param_range, testing_acc, label= 'Testing accuracy', lw= 2, linestyle= '--', marker= 'o', color= 'red')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy over a variety of {param_name} values')
    plt.xticks(x_ticks)
    plt.legend()
    plt.savefig('Results/ParameterPlots/' + model_name + '_' + param_name)
    plt.show()


def plot_mean_feature_values(filepath, scaled= False):
    data = pd.read_csv(filepath)
    if not scaled:
        data = data.drop(["filename", "length"], axis=1)
    mean_features = data.groupby('label').mean().reset_index()

    melted_df = pd.melt(mean_features, id_vars=["label"], var_name="Feature", value_name="Mean")

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.barplot(data=melted_df, x='Feature', y='Mean', hue='label')
    plt.title('Mean Feature Values by Genre')
    plt.xticks(rotation=45)  # Rotates the feature names for better readability
    plt.ylabel('Mean Value')
    plt.xlabel('Feature')
    plt.legend(title='Genre')
    plt.show()

def plot_train_test_accuracies(train, test, index, path):
    plt.figure(figsize=(10, 6))
    plt.plot(index, train, label='Training accuracy', color= 'purple')
    plt.plot(index, test, label= 'Testing accuracy', color= 'green')
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of features')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    #plot_model_parameter_range(model_name= "KNN", param_name= "n_neighbors", param_range=range(1, 42), x_ticks= [1, 9, 17, 25, 33, 41])
    plot_mean_feature_values('Datasets/features_30_sec.csv')
    plot_mean_feature_values('Datasets/features_30_sec_scaled.csv', scaled= True)





































""" def plot_gridsearch_results(gridsearch, model_name, param_name):
    #Plots the accuracy changing as the gridsearch iterates trough different values for a certain parameter

    results = pd.Dataframe(gridsearch.cv_results_)
    param_column = f'param_{param_name}'

    if param_column in results:
        mean_scores = results.groupby(param_column).apply(
            lambda x: pd.Series({
                'mean_train_score': x['mean_train_score'].mean(),
                'mean_test_score': x['mean_test_score'].mean()

            })
        ).reset_index()
    
    plt.plot(fig_size = (12, 10))
    plt.plot(mean_scores[param_column], mean_scores['mean_train_score'], label= 'Mean Train Score')
    plt.plot(mean_scores[param_column], mean_scores['mean_test_score'], label= 'Mean Test Score')
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.title(f"Test vs Training accuracy for {model_name} with varying {param_name}")
    plt.legend()
    plt.save('Results/TrainingPlots/' + model_name + '_' + param_name)
    plt.show() """

