import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string
from sklearn.utils import shuffle
from nltk.corpus import stopwords

from src.logger import logging
from src.exception import CustomException
from nltk.stem.snowball import SnowballStemmer


df = pd.read_csv('artifacts/train.csv',low_memory=False)


# stop_words = stopwords.words("english")
# wordnet = WordNetLemmatizer()


class DataCleaner:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')

    def clean_text(self, text):
        """Cleans a text string using NLTK.

        Args:
            text: A string containing the text to be cleaned.

        Returns:
            A string containing the cleaned text.
        """

        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # Remove stop words
        tokens = [token for token in tokens if token not in self.stopwords]

        # Convert all words to lowercase
        tokens = [token.lower() for token in tokens]

        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]

        # Remove any remaining empty tokens
        tokens = [token for token in tokens if token]

        # Return the cleaned text
        return ' '.join(tokens)

    def clean_dataframe(self, dataframe, columns=None):
        """Cleans a Pandas DataFrame using NLTK.

        Args:
            dataframe: A Pandas DataFrame containing the text to be cleaned.
            columns: A list of column names to clean, or None to clean all columns.

        Returns:
            A Pandas DataFrame containing the cleaned text.
        """

        if columns is None:
            columns = dataframe.columns

        for column in columns:
            dataframe[column] = dataframe[column].apply(self.clean_text)

        return dataframe

cleaner = DataCleaner()
a = df[['_source.issue']]
a = cleaner.clean_dataframe(a)
# print(df.head())
print(a.head())
