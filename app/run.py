import json
import plotly
import pandas as pd
import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

def tokenize(text):
    """
    Tokenizing and lemmatizing, made available for the pickled model
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Retrieve whether or not a message starts with a verb, made available for the pickled model
    """

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
    """
    Retrieve the length of the message, made available for the pickled model
    """
    def message_length(self, messages):
        for message in messages:
            return len(message)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_measured = pd.Series(X).apply(self.message_length)
        return pd.DataFrame(X_measured)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster-messages', engine)

# load model
model = joblib.load("../models/finalized_model.pkl")

# extract data needed for visuals
# TODO: Below is an example - modify to extract data for your own visuals
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

## Get variables for category graph
category_counts = df.columns[4:39].to_series()
category_counts.values[:] = 0
for i in range(0, df.shape[0]-1):
   for category in category_counts.index:
       if df.iloc[i][category] == 1:
           category_counts[category] += 1
                
category_names = list(category_counts.index)

## Get variables for category message length graph
    
df_message_lengths = df
df_message_lengths['length'] = df_message_lengths.message.apply(lambda x: len(x))

category_lengths = df.columns[4:39].to_series()
category_lengths.values[:] = 0

for i in range(0, df.shape[0]-1):
    for category in category_lengths.index:
        if df.iloc[i][category] == 1:
            category_lengths[category] += df_message_lengths.iloc[i]['length']

for category in category_lengths.index:
    if category_counts[category] != 0:
        category_lengths[category] = category_lengths[category]/category_counts[category]
    else:
        category_lengths[category] = 0


category_lengths_names = list(category_lengths.index)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Creates graphs to be made availble as variables to the html template
    """

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_lengths_names,
                    y=category_lengths
                )
            ],

            'layout': {
                'title': 'Distribution of Message Lengths',
                'yaxis': {
                    'title': "Average Message Length"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Defines the routing for the message query action, and which variables will be passed on to the template.
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Hosts the flask application at the defined values below.
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()