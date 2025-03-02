import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """ Tokenize input messages
    Args:
        text: messages to classify
    Returns:
        clean_tokens: messages have been normalized, tokenized, and lemmatized
    """
    
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    
    # tokenize text 
    tokens = word_tokenize(text)
    
    # initate WordnetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize, remove stopwords, normalize and remove blank space before and after the tokens
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok not in stopwords.words('english')]
    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('processed_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # To get TOP 10 category name count
    top_category_count = df.iloc[:,4:].sum().sort_values(ascending=False)[1:11]
    top_category_names = list(top_category_count.index)
    
    # create visuals
    graphs = [
        {
            "data": [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {
                    "title": "Count"
                },
                "xaxis": {
                    "title": "Genre"
                }
            }
        },

        {
            "data": [
                Bar(
                    x=top_category_names,
                    y=top_category_count
                )
            ],

            "layout": {
                "title": "Top Ten Categories",
                "yaxis": {
                    "title": "Count"
                },
                "xaxis": {
                    "title": "Categories"
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
