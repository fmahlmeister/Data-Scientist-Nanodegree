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

	# load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, con=engine)
	X = df.loc[:,'message']
	Y = df.iloc[:,4:]
	category_names = Y.columns

	return X, Y, category_names


def tokenize(text):
    # normalize case and remove punctuation    
    tokenizer = RegexpTokenizer(r'\w+')

    # tokenize text
    token_words = tokenizer.tokenize(text.lower())
    
    # remove stop words
    token_words = [w for w in token_words if w not in stopwords.words('english')]
    
    # extract root form of words
    token_words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in token_words]
    
    return token_words


def save_stats(X, Y, category_names, vocabulary_stats_filepath, category_stats_filepath):
    
    """Save stats
    Args;
        X: numpy.ndarray. Disaster messages.
        Y: numpy.ndarray. Disaster categories for each messages.
        category_names: Disaster category names.
        vocaburary_stats_filepath: String. Vocaburary stats is saved as pickel into this file.
        category_stats_filepath: String. Category stats is saved as pickel into this file.
    """

    # Check vocabulary
    vect = CountVectorizer(tokenizer=tokenize)
    X_vectorized = vect.fit_transform(X)

    # Convert vocabulary into pandas.dataframe
    keys, values = [], []
    for k, v in vect.vocabulary_.items():
        keys.append(k)
        values.append(v)
    vocabulary_df = pd.DataFrame.from_dict({'words': keys, 'counts': values})

    # Vocabulary stats
    vocabulary_df = vocabulary_df.sample(30, random_state=42).sort_values('counts', ascending=False)
    vocabulary_counts = list(vocabulary_df['counts'])
    vocabulary_words = list(vocabulary_df['words'])

    # Save vocaburaly stats
    with open(vocabulary_stats_filepath, 'wb') as vocabulary_stats_file:
        pickle.dump((vocabulary_counts, vocabulary_words), vocabulary_stats_file)

    # Category stats
    category_counts = list(Y.sum(axis=0))

    # Save category stats
    with open(category_stats_filepath, 'wb') as category_stats_file:
        pickle.dump((category_counts, list(category_names)), category_stats_file)



def build_model():

	pipeline = Pipeline([
		('tfidfV', TfidfVectorizer(tokenizer=tokenize)),
		('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
	])

	# parameters = {'tfidfV__ngram_range': ((1, 1), (1, 2)),
	#               'tfidfV__max_df':(0.75, 1.0),
	#               'clf__estimator__estimator__C':[0.5,1], 
	#               'clf__estimator__estimator__max_iter':[100]}

	parameters = {}

	# Construct grid searches
	jobs = 1

	# Create model
	model = GridSearchCV(pipeline,param_grid=parameters,verbose=2,n_jobs=jobs,cv=2)

	return model


def evaluate_model(model, X_test, Y_test, category_names):
    
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
    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 5:
        database_filepath, model_filepath, vocabulary_filepath, category_filepath  = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Saving stats...')
        save_stats(X, Y, category_names, vocabulary_filepath, category_filepath)
        
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
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl '\
              'vocabulary_stats.pkl category_stats.pkl')


if __name__ == '__main__':
    main()