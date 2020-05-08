import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Reads the messages and the categories dataset and merges them

    Input:
        messages_filepath (str): location of the messages dataset
        categories_filepath (str): location of the categories dataset

    Returns:
        pd.DataFrame
    """

    # Read datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = messages.merge(categories, 'inner', 'id')

    return df


def clean_data(df):
    """Creates dummy variables from the categories column and drops
    duplicates

    Input:
        df (pd.DataFrame): raw dataset with messages and categories

    Returns:
        pd.DataFrame
    """

    # Create dummy representation of categories in new dataset
    categories = df.categories.str.split(';', expand=True)
    categories.columns = categories.iloc[0].str.split('-').str[0]
    categories = categories.apply(lambda s: s.str.split('-').str[1].astype(int))

    # Replace raw categories column in df with dummy representation
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates from df
    df = df.drop_duplicates('id')

    return df


def save_data(df, database_filename):
    """Loads dataset into local database using sqlite

    Input:
        df (pd.DataFrame) - dataset to be loaded
        database_filename (str) - database filename

    Returns:
        None
    """

    # Create engine
    engine = create_engine('sqlite:///{}'.format(database_filename))

    # Load data into database
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()