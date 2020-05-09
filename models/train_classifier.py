import sys
import pandas as pd
from sqlalchemy import create_engine
import string
import pickle
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """Loads dataset from database and splits data into feature
    and target set

    Input:
        database_filepath (str): location of the local database

    Returns:
        X (np.array): raw messages
        Y (np.array): category dummies
        category_names (list): category labels
    """

    # Load dataset from local database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)

    # Create feature and target set
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values

    # Save category names in list
    category_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """Simple tokenizer that cleans a given text and splits it into individual
    tokens. The cleaning process includes the removal of punctuation, lowercasing
    and lemmatization.

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
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    return lemmas


def build_model():
    """Create a machine learning model pipeline that first transforms input
     documents into a tf-idf representation and then applies a multi class
     classification using a random forest

     Returns:
         sklearn.model_selection._search.GridSearchCV: estimator
     """

    # Create text classification pipeline
    pipeline = Pipeline([
        ('counts', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Create parameter space for hyperparameter tuning
    parameters = {
        'tfidf__smooth_idf': [False, True],
        'clf__estimator__n_estimators': [100, 200, 300, 400, 500],
        'clf__estimator__max_features': ['sqrt', 'log2']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Uses model to make predictions on the test set and prints
    classification report for each category

    Input:
        model (estimator object): multiclass classifier
        X_test (np.array): test set features
        Y_test (np.array): test set categories
        category_names (list): category names

    Output:
        None
    """

    # Make predictions on test set
    Y_pred = model.predict(X_test)

    # Print classification report for each category
    for i, category_name in enumerate(category_names):
        print(category_name)
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """Saves model as a pickle file

    Input:
        model (estimator object): multiclass classifier
        model_filepath (str): storage location for model

    Returns:
        None
    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print(X)
        print(Y)
        print(category_names)
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