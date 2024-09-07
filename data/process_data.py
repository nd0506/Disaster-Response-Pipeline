# Import libraries
import sys
import sqlite3
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

# load data
def load_data(messages_filepath, categories_filepath):
    """ load messages and categories data
    Args:
        messages_filepath = file path of messages data
        categories_filepath = file path of categories data
    Returns: 
        df : merged data of messages and categories
    """
    # load messages data
    messages = pd.read_csv(messages_filepath)
    # load categories data
    categories = pd.read_csv(categories_filepath)
    # merge messages and categories using id as inner join key 
    df = pd.merge(messages, categories, on = 'id', how = 'inner')
    return df
   
def clean_data(df):
    """ Clean merged data of messages and categories
    Args:
        df: merged dataframe of messages and categories after loaded
    Returns: 
        df : clean dataframe after transforming categories, deduplication,..
    """
    # create new categories columns by splitting old categories column
    categories = df['categories'].str.split(';', expand = True)
    
    # create new head for categories 
    # by using the first string before '-' of values in the first row
    new_header = categories.iloc[0].astype(str).apply(lambda x: x.split('-')[0])
    
    # apply new column names
    categories.columns = new_header
    
    # convert values in categories columns to int by
    # looping through each column and take the number after '-', then covnerting them to int
    for column in categories.columns:
        categories[column] = categories[column].astype(str).str.split('-').str[1]
        categories[column] = categories[column].astype(int)
        
    # drop the old categories column
    df = df.drop(columns=['categories'])
    
    # concat new categories columns with inital dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates where messages have the same id
    df  = df.drop_duplicates(subset=['id'])
    
    # Assert that there are no duplicated rows after deduplication
    assert len(df[df.duplicated()]) == 0
    
    # clean df by filtering out messages has value: #NAME? and unreasonable value in column related
    df = df[(df['message']!='#NAME?') & (df['related'] !=2)]
    
    return df


def save_data(df, database_filename):
    """ Save clean df to sql database and table
    Args:
        df: clean dataframe
        database_filename: name of database
    Returns:
        data that saved in process_message table
    """
    # create database
    engine = create_engine(f'sqlite:///{database_filename}')
    # load df to sql table
    df.to_sql('processed_messages', engine, if_exists = 'replace', index=False)

              

def main():
    """ Run above 3 functions
    """
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
