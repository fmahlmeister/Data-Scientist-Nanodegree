# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
  # load messages dataset
  messages = pd.read_csv(messages_filepath)

  # load categories dataset
  categories = pd.read_csv(categories_filepath)

  # merge datasets
  df = pd.merge(left=messages, right=categories, on='id', how='left')

  return df



def clean_data(df):

  # create a dataframe of individual category columns  
  categories = df.categories.str.split(';',expand=True)

  # select the first row of the categories dataframe
  row = categories.iloc[1,:]

  # extract a list of new column names for categories.
  category_colnames = row.apply(lambda x: x.split('-')[0])

  # rename the columns of `categories`
  categories.columns = category_colnames

  for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str.split('-', expand=True)[1]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

  # drop the original categories column from `df`
  df.drop(columns='categories', inplace=True)

  # concatenate the original dataframe with the new `categories` dataframe
  df = pd.concat([df,categories], axis=1)

  # drop duplicates
  df.drop_duplicates(inplace=True)

  return df



def save_data(df, database_filename, table_name='DisasterResponse'):

  engine = create_engine('sqlite:///' + database_filename)
  df.to_sql(table_name, engine, index=False, if_exists='replace', chunksize=600)



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