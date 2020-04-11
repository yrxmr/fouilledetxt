import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
nltk.download('wordnet')
import pickle
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

corpus_data = load_files(r"/home/y/Documents/corpus")
X, y = corpus_data.data, corpus_data.target

documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
predicted = rf_clf.predict(X_test)
print(np.mean(predicted == y_test))

nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)
predicted = nb_clf.predict(X_test)
print(np.mean(predicted == y_test))

sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train)
predicted = sgd_clf.predict(X_test)
print(np.mean(predicted == y_test))

kn_clf = KNeighborsClassifier()
kn_clf.fit(X_train, y_train)
predicted = kn_clf.predict(X_test)
print(np.mean(predicted == y_test))

lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
predicted = lr_clf.predict(X_test)
print(np.mean(predicted == y_test))




