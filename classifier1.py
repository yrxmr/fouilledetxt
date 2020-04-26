#le script est à exécuter sur un dossier comprenant des sous-dossiers, chacun correspondant à une catégorie et comprenant
#les fichiers TXT de la catégorie donnée 

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

#avec load_files on charge les fichiers, chaque catégorie correspond à un sous-dossier
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
#indice min df= 5, le mot en question doit être trouvé dans au moins 5 documents pour être pris en compte
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

#on assigne un score quant au nombre d'occurrences des mots avec l'indice TFIDF (term frequency/inverse document frequency)
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

#division en "training set" et "test set"
#25% des données serviront d'entraînement du modèle, tandis que les prédictions seront
#effectuées sur les 75% restants
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=0)


#on charge le modèle
nb_clf = MultinomialNB()
#fit= on applique le modèle aux données d'entraînement
nb_clf.fit(X_train, y_train)
#on effectue des prédictions sur les données constituant le reste du corpus
predicted = nb_clf.predict(X_test)
#on obtient un indice de succès général
print("Accuracy for the Multinomial NB:")
print(np.mean(predicted == y_test))
#une matrice de confusion
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))
#les détails de la classification(p,r,f-mesure)
print("Classification Report:")
print(classification_report(y_test,predicted))


#on répète les mêmes opérations pour les autres modèles
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
