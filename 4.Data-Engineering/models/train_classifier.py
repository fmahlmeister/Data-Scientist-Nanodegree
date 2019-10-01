# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def load_data(database_filepath, table_name='DisasterResponse'):

    """
    Load cleaned data from database into dataframe

    Parameters
    ----------
    database_filepath: str
        It contains cleaned data table
    table_name: str (optional)
        It contains cleaned data

    Returns
    -------
    X: numpy.ndarray
        Disaster messages
    Y: numpy.ndarray
        Disaster categories for each messages.
    category_name: list of str
        Disaster category names.
    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, con=engine)
    X, Y = df.loc[:,'message'], df.iloc[:,4:]
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):

    """
    Tokenize text (a disaster message)

    Parameters
    ----------
    text: str
        A disaster message

    Returns
    -------
    token_words: list of int
        It contains tokens
    """

    # normalize case and remove punctuation    
    tokenizer = RegexpTokenizer(r'\w+')

    # tokenize text
    token_words = tokenizer.tokenize(text.lower())
    
    # remove stop words
    token_words = [w for w in token_words if w not in stopwords.words('english')]
    
    # extract root form of words
    token_words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in token_words]
    
    return token_words


def build_model():

    """
    Build model

    Parameters
    ----------

    Returns
    -------
    model: sklearn.model_selection.GridSearchCV
    It contains a sklearn estimator
    """

    pipeline = Pipeline([
    ('tfidfV', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])

    parameters = {'tfidfV__ngram_range': ((1, 1), (1, 2)),
                  'tfidfV__max_df':(0.75, 1.0),
                  'clf__estimator__estimator__C':[0.5,1], 
                  'clf__estimator__estimator__max_iter':[100]}

    # Construct grid searches
    jobs = 1

    # Create model
    model = GridSearchCV(pipeline,param_grid=parameters,verbose=2,n_jobs=jobs,cv=2)

    return model


def evaluate_model(model, X_test, Y_test, category_names):

    """
    Evaluate model

    Parameters
    ----------
    model : sklearn.model_selection.GridSearchCV
        It contains a sklearn estimator
    X_test : numpy.ndarray
        Disaster messages
    Y_test : numpy.ndarray
        Disaster categories for each messages
    category_names : list of str
        Disaster category names

    Returns
    -------

    """
    
    # Predict on test data with best params
    Y_pred = model.predict(X_test)

    f1_list = []
    categories_name = list(Y_test.columns)
    
    for i in range(0, len(Y_test.columns)):
 
        f1_list.append(f1_score(Y_test.values[:, i], Y_pred[:, i], average='weighted'))
    
    df = pd.DataFrame([f1_list],columns=categories_name,index=['f1_score_1'])

    df = df.T

    # Test data accuracy of model with best params
    # f1_score = multioutput_classification_report(Y_test,Y_pred).f1_score.mean()
    print('Test f1 score for best params: %.3f ' % df.f1_score_1.mean())


def save_model(model, model_filepath):

    """
    Save the model

    Parameters
    ----------
    model : sklearn.model_selection.GridSearchCV
        It contains a sklearn estimator
    model_filepath : str
        Trained model is saved as pickle into this file
        
    Returns
    -------

    """

    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath  = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
      
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'models/train_classifier.py ../data/DisasterResponse.db models/classifier.pkl')


if __name__ == '__main__':
    main()