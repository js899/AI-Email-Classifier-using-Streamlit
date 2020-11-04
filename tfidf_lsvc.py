from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pickle

def tfidf_lsvc_train(df_to_use):
    X_train, X_test, y_train, y_test = train_test_split(df_to_use['Consumer_Complaint'], df_to_use['Product'], random_state = 42)
    cv = TfidfVectorizer(max_features=840, min_df=4,  max_df=0.7)
    features = cv.fit_transform(X_train)
    clf = LinearSVC().fit(features, y_train)
    # For Later Testing
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    # Quick Testing
    y_pred = clf.predict(cv.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Testing function for new data
def tfidf_lsvc_test(test_data):
    #X_test = test_data[1][0]
    #loaded_model = pickle.load(open(filename, 'rb'))
    #y_pred = loaded_model.predict(X_test)
    #result = loaded_model.score(y_pred, y_test)
    return #accuracy