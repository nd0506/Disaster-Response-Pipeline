#import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import pickle
from sklearn.model_selection import GridSearchCV
nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """ Load X, y factors and categories columns for later model report
    Args:
        database_filepath: database name that contain process_message table
    Returns:
        X: values in message column
        y: values in categories columns
        category_names: column names of all categories
    """
    # create connection to database
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # load process_message table
    df = pd.read_sql('processed_messages', con=engine)
    
    # select X factor
    X = df['message'].values
    
    # select Y factor: all columns from column number 4th
    y = df.iloc[:, 4:].values
    
    # retrieve categories column name
    category_names = df.iloc[:, 4:].columns
    return X, y, category_names


def tokenize(text):
    """ Tokenize messages
    Args:
        text: messages
    Returns:
        clean_tokens: texts have been normalized, tokenized, 
                      lemmatize removed punctuation, and stopwords
    """
    # remove punctuation 
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    
    # tokenize text 
    tokens = word_tokenize(text)
    
    # initate WordnetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize, remove stopwords, normalize and remove blank space before and after the tokens
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok not in stopwords.words('english')]
    return clean_tokens


def build_model():
    """ Build ML model
    Returns: 
        Model: Machine learning model to classify messages into different categories
        This model is using gridsearch to find the best parameters  
    """
    # create ML pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # create list of searching parameters
    parameters = {
        'vect__ngram_range': [(1, 1)],  
        'clf__estimator__n_estimators': [50], 
        'clf__estimator__min_samples_split': [2]
    }

    # use grid search with ML pipeline to find the best parameters
    model = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=3, n_jobs=-1)

    return model


def evaluate_model(model, y_test, y_pred, category_names):
    """ Evaluate model
    Args:
        model: defined NLP model in previous function
        y_test: categories data for testing
        y_pred: predicted categories data
        categories_names: names of categories columns
    Returns: 
       evaluation metrics with confusion matrix, accuracy for each category
    """
    overall_accuracy = (y_pred == y_test).mean()
    print(f"Overall Accuracy: {overall_accuracy:.4f}\n")
    print("Best Parameters:", model.best_params_, "\n")
    
    # Loop through each column (category) and print classification report and confusion matrix
    for i, col in enumerate(category_names):
        print(f"Category: {col}")
        labels = np.unique(y_test[:, i])
        confusion_mat = confusion_matrix(y_test[:, i], y_pred[:, i], labels=labels)
        accuracy = (y_pred[:, i] == y_test[:, i]).mean()

        print("Labels:", labels)
        print("Confusion Matrix:\n", confusion_mat)
        print("Accuracy:", accuracy)
        
        # Print classification report for each category
        print("Classification Report:\n", classification_report(y_test[:, i], y_pred[:, i], labels=labels))
        print("-" * 60)

def save_model(model, model_filepath):
    """ Save model as pickle file
    Args:
        model: trained model
        model_filepath: path of saved model
    Returns: 
        model saved in given file path
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """ Execute load data, model training, model report and save model
    Returns: 
        model is trained and saved as classifier.pkl
    """
    # load data and split data
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # initiate model
        print('Building model...')
        model = build_model()
        
        # train model
        print('Training model...')
        model.fit(X_train, y_train)
        
        # use model to predict test data
        y_pred = model.predict(X_test)
        
        # Evaluate model
        print('Evaluating model...')
        evaluate_model(model, y_test, y_pred, category_names)
        
        # Save model
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