import json
import plotly
import pandas as pd
import string
import nltk
from collections import Counter
nltk.download(['wordnet', 'stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    """Simple tokenizer that cleans a given text and splits it into individual
    tokens. The cleaning process includes the removal of punctuation and stopwords,
    lowercasing and lemmatization.

    Input:
        text (str): raw text

    Returns:
        list: tokens
    """

    # Replace punctuation with whitespace
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    # Split text into words by whitespace
    tokens = text.lower().split()

    # Apply lemmatization to words
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token.strip()) for token in tokens
              if token not in set(stopwords.words('english'))]

    return lemmas


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # count genres
    genre_counts = df.genre.value_counts()
    genre_names = list(genre_counts.index)

    # count categories
    category_counts = df.iloc[:,4:].sum(axis=0).sort_values(ascending=False)
    category_names = list(category_counts.index)

    # create visuals
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
                 'title': 'Distribution of Categories',
                 'yaxis': {
                     'title': "Count"
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()