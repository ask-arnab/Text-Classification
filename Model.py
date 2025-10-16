#Taking Imports
from Text_classification import *
from Chat_treatment import *
import pandas as pd
import time
import matplotlib.pyplot as plt
#Loading the dataset
df = pd.read_csv("IMDB Dataset.csv")

#Creating a smaller dataset for testing purposes

df = df.sample(10000, random_state=42)

#Done Basic EDA and Preprocessing
df = Text_Prepressing(df)

#Taking imports for model building
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Defining X&Y
X = df.iloc[:,0]
Y = df.iloc[:,1]

#Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=42)

#Label Encoding
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

#Appling Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_bow = cv.fit_transform(X_train).toarray()
X_test_bow = cv.transform(X_test).toarray()
#Creating the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
#Training the model
print("Model training started.......................................")
start = time.time()
model.fit(X_train_bow, Y_train)
print("Training Completed")
print("Time taken for training:",time.time()-start)
print("Enering into evaluation phase...")
#Evaluating the model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
Y_pred = model.predict(X_test_bow)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n")
print(classification_report(Y_test, Y_pred))

#Confusion Matrix
from mlxtend.plotting import plot_confusion_matrix
plot_confusion_matrix(confusion_matrix(Y_test, Y_pred), figsize=(6,6),colorbar=True, class_names=['negative','positive'])
plt.show()