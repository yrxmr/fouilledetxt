import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#avec load_files on convertit les fichiers en données traitables

corpus_data = load_files(r"~/Documents/corpus")
X, y = corpus_data.data, corpus_data.target

documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):

    #on retire les noms des catégories comme pour les expérimentations de Weka
    document = re.sub(r'romantic|romance|family|families|party|parties|adventure|adventures', '', str(X[sen]))
    
    #on effectue la lemmatisation grâce à NLTK
    document = document.lower()
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)
    
#traitement des données textuelles pour qu'elles puissent être traitées
#utilisation de "bag of words" pour convertir des données textuelles en données numériques, en ignorant les stopwords
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

#on assigne un score quant au nombre d'occurrences des mots avec l'indice TFIDF (term frequency/inverse document frequency)
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

#division en training set et test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=0)

#on applique les import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
nltk.download('wordnet')
import pickle
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


corpus_data = load_files(r"/home/y/Documents/corpus")
X, y = corpus_data.data, corpus_data.target

documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):

    document = re.sub(r'romantic|romance|family|families|party|parties|adventure|adventures', '', str(X[sen])
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=0)


nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)
predicted = nb_clf.predict(X_test)
print("Accuracy for the Multinomial NB:")
print(np.mean(predicted == y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))
print("Classification Report:")
print(classification_report(y_test,predicted))



tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)
predicted = tree_clf.predict(X_test)
print("Accuracy for the Decision Tree Classifier:")
print(np.mean(predicted == y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))
print("Classification Report:")
print(classification_report(y_test,predicted))


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
predicted = rf_clf.predict(X_test)
print("Accuracy for the Random Forest Classifier:")
print(np.mean(predicted == y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))
print("Classification Report:")
print(classification_report(y_test,predicted))différents modèles
#on obtient un score général, une matrice de confusion
et des données plus détailles(P,R,F-mesure etc.)

#on charge le modèle
nb_clf = MultinomialNB()
#fit= on applique le modèle aux données
nb_clf.fit(X_train, y_train)
predicted = nb_clf.predict(X_test)
print("Accuracy for the Multinomial NB:")
print(np.mean(predicted == y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))
print("Classification Report:")
print(classification_report(y_test,predicted))



tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)
predicted = tree_clf.predict(X_test)
print("Accuracy for the Decision Tree Classifier:")
print(np.mean(predicted == y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))
print("Classification Report:")
print(classification_report(y_test,predicted))


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
predicted = rf_clf.predict(X_test)
print("Accuracy for the Random Forest Classifier:")
print(np.mean(predicted == y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))
print("Classification Report:")
print(classification_report(y_test,predicted))
