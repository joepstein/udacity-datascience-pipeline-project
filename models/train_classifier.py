import sys

from sqlalchemy import create_engine

import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator, TransformerMixin

import pickle


def load_data(database_filepath):
    """
    Retrieve the data for the model, from the specified location
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster-messages', engine)
    X = df.message.values
    Y = df.drop(columns=['id', 'message', 'original', 'genre']) 
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Leverage NLTK for tokenization. Unify the words over the lemmatized version of the word as well.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Create a pipeline to train the model on classification of the messages.
    Engineer features such as StartingVerbExtractor, and MessageLengthExtractor
    """
    class StartingVerbExtractor(BaseEstimator, TransformerMixin):

        def starting_verb(self, text):
            sentence_list = nltk.sent_tokenize(text)
            for sentence in sentence_list:
                pos_tags = nltk.pos_tag(tokenize(sentence))
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            return False

        def fit(self, x, y=None):
            return self

        def transform(self, X):
            X_tagged = pd.Series(X).apply(self.starting_verb)
            return pd.DataFrame(X_tagged.astype(float))

    class MessageLengthExtractor(BaseEstimator, TransformerMixin):
        def message_length(self, messages):
            for message in messages:
                return len(message)

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X_measured = pd.Series(X).apply(self.message_length)
            return pd.DataFrame(X_measured)

    pipeline = Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),

                ('message_length', MessageLengthExtractor()),
                ('starting_verb', StartingVerbExtractor())
            ])),
            ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
        ])

    parameters = {
            'features__text_pipeline__vect__max_df': (0.8, 1),
            'features__text_pipeline__tfidf__smooth_idf': (True, False),
            "clf__estimator": [RandomForestClassifier()],
            'clf__estimator__n_estimators': [100, 200],
            'clf__estimator__criterion': ['gini', 'entropy'],
            'features__transformer_weights': (
                { 'text_pipeline': 1, 'starting_verb': 0.5, 'message_length' : 0.2 },
                { 'text_pipeline': 1, 'starting_verb': 1, 'message_length' : 0.7 },
            )
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model on each category, and output the results
    """
    y_pred = model.predict(X_test)

    for index, column in enumerate(y_test):
        print(category_names[index])
        print(classification_report(y_test.iloc[index], y_pred[index]))


def save_model(model, model_filepath):
    """
    Pickle the model for reuse in the flask app
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Load the database, and pipe the data through different functions to test classification
    The model must then be stored for the flask app.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()