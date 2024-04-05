import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import string
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  confusion_matrix
from sklearn.model_selection import train_test_split

#this is the model class which is used for training and testing the data
class NaiveBayesMulticlassClassifier:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_words = defaultdict(int)
        self.class_prior = defaultdict(int)

    #fucntion to preprocess the text using different preprocessing methods as below in the function
    def preprocess_text(self, text):
        # Tokenization
        tokens = word_tokenize(text.lower())
        # Here stopwords and punctuation are removeed and lemmatize is applied to each document/sentence
        stop_words = set(stopwords.words('english'))
        punctuation = set(string.punctuation)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in punctuation]
        return tokens

    #This function is reponsible for training the data.
    #It calculates count of each word belong to that class. Each word count is stored in a dictionary of dictionary
    def train(self, train_data, train_labels):
        for doc, label in zip(train_data, train_labels):
            drug_name, condition, review = doc
            words = self.preprocess_text(drug_name + ' ' + condition + ' ' + review)
            for word in words:
                self.word_counts[label][word] += 1
                self.total_words[label] += 1
        #calculating the numbe of label for finding the prior probabilities in the next step        
        num_documents = len(train_labels)
        for label, word_count in self.word_counts.items():
            self.class_prior[label] = len([l for l in train_labels if l == label]) / num_documents
    
    #This function does the classification by using the above trained data
    def classify(self, document):
        drug_name, condition, review = document
        words = self.preprocess_text(drug_name + ' ' + condition + ' ' + review)
        scores = {label: np.log(self.class_prior[label]) for label in self.class_prior.keys()}
        #finding the probability for each model and storing it in scores dictionart
        for label, word_count in self.word_counts.items():
            for word in words:
                scores[label] += np.log((word_count[word] + self.alpha) / (self.total_words[label] + self.alpha * len(word_count)))
        #comparing the probabilities and choosing the model with max probability as predicted label
        predicted_label = max(scores, key=scores.get)
        return scores, predicted_label

#Start the classification:
if __name__ == "__main__":

    #reading the parameters from command line
    if len(sys.argv) != 2 or float(sys.argv[1]) < 20 or float(sys.argv[1]) > 80 or not sys.argv[1].isdigit(): 
        print("Assuming TRAIN_SIZE is 80.")
        train_percent = 80
    else:
        train_percent = float(sys.argv[1])
    # Read train data percentage from command-line argument


    # Load dataset (replace 'dataset.csv' with your file path)
    df = pd.read_csv("UCIdrug_train.csv")

    df = df.head(50000)
    # Drop rows with missing values
    df.dropna(subset=['drugName', 'condition', 'review', 'rating'], inplace=True)

    # Extract necessary columns
    train_data = df[['drugName', 'condition', 'review']].values
    
    train_labels = df['rating'].values  # Assuming 'rating' is the column for labels

    print(f"The model is training on  : {train_percent}  percentage train data" )


    # Split data into training and testing sets using the passed train_perecent variable    
    # Calculate the index to split the data based on the user-defined training percentage
    train_index = int(len(train_data) * (train_percent / 100))

    # Calculate the index to split the data based on the last 20% for the test set
    test_index = int(len(train_data) * 0.8)

    # Split data into training and testing sets
    X_train, X_test = train_data[:train_index], train_data[-test_index:]
    y_train, y_test = train_labels[:train_index], train_labels[-test_index:]

    
    # Initialize and train classifier
    print("Training the classifier.........")
    classifier = NaiveBayesMulticlassClassifier()
    classifier.train(X_train, y_train)


    print("Testing the classififer........")
   # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Classify each sample in the test set and collect true labels and predicted labels
    for sample, true_label in zip(X_test, y_test):
        result = classifier.classify(sample)
        #classifier return twp things : one is conditional proababi.ity and other is predicted label which is decided from max of the probabilities
        scores, predicted_label = result
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
    
    print("Test results....") 
    # Compute confusion matrix : The confusion matrix is a K×K matrix, where  K is the number of classes, representing the counts of true positive, false positive, true negative, and false negative predictions for each 
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    #Extract and calcaulte TP, TN, FP, FN from the confusion matrix
    TP = np.diag(conf_matrix)
    TN = np.sum(conf_matrix) - np.sum(conf_matrix, axis=1) - np.sum(conf_matrix, axis=0) + 2*TP
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP

    # Print results
    for i in range(len(conf_matrix)):
        print(f"Class {i+1}:")
        print(f"True Positives (TP): {TP[i]}")
        print(f"True Negatives (TN): {TN[i]}")
        print(f"False Positives (FP): {FP[i]}")
        print(f"False Negatives (FN): {FN[i]}")


    # Calculate Average True Positive (ATP)
    ATP = np.mean(TP)
    # Calculate Average True Negative (ATN)
    ATN = np.mean(TN)
    # Calculate Average False Positive (AFP)
    AFP = np.mean(FP)
    # Calculate Average False Negative (AFN)
    AFN = np.mean(FN)

    # Print average values
    print(f"Average True Positive (ATP): {ATP}")
    print(f"Average True Negative (ATN): {ATN}")
    print(f"Average False Positive (AFP): {AFP}")
    print(f"Average False Negative (AFN): {AFN}")


    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Calculate precision
    precision = precision_score(true_labels, predicted_labels, average='macro')

    # Calculate recall
    recall = recall_score(true_labels, predicted_labels, average='macro')

    # Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    # Calculate specificity
    TN = conf_matrix[0, 0]  # True negatives
    FP = conf_matrix[0, 1]  # False positives

    specificity = TN / (TN + FP)
    print("Specificity:", specificity)

    # specificity = specificity_score(true_labels, predicted_labels)
    print("Sensitivity(Recall):", recall)
    print("Specificity:", specificity)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("F-Score:", f1)
    

#Clasifying based on the user input
#Loop for classifying sentences
while True:
    # Input sentence from the user
    print("\n")
    drugName = input("Enter drug name:")
    condition = input("Enter condition:")
    review = input("Enter review:")
    
    
    # Each user input is Classified using the already trained classifier, so no more trainn
    result = classifier.classify((drugName, condition , review))
    scores, predicted_label = result
    
    print(f"Entered review : {review} for drug s: {drugName}")
    print(f"was classified as class {predicted_label}")
    

    # Convert log probabilities to probabilities
    probabilities = {label: np.exp(log_prob) for label, log_prob in scores.items()}
    print(f"\nProbabilities of this document belong to each class")
    
    for key, value in probabilities.items():
        print(f"P({key} | {review}) = {value}")
    
    # Ask if the user wants to enter another sentence
    choice = input("Do you want to enter another sentence [Y/N]? ")
    if choice.lower() != 'y':
        break


# #***********************************************************************
#commenting the code to show number of samples    
# total_samples = len(df)
# print("Total number of samples:", total_samples)

# # Get the counts of samples for each label
# label_counts = df['rating'].value_counts().sort_index()

# # Plot the bar chart
# plt.bar(label_counts.index, label_counts.values)
# plt.xlabel('Rating')
# plt.ylabel('Number of Samples')
# plt.title('Number of Samples per Label')
# plt.show()


# print(f"total number of labels: {len(classifier.word_counts)}")
# print('\n')

# print("Displaying the number of samples per label")
# for label, count in label_counts.items():
#     print(f"Label {label}: {count} samples")

# print("\nnumber of words in each class-dictionary")
# for label, words_list in classifier.word_counts.items():
#     print(f"Class {label}: {len(words_list)} words")

# #*********************************************************************************
# # Example usage:
# drug_name = "Ortho Evra"
# condition = "stomachache"
# review = "This drug is very effective!"
# predicted_rating = classifier.classify((drug_name, condition, review))
# print("Predicted rating:", predicted_rating)

