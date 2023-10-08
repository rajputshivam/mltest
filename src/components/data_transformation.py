import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import re
import string
from sklearn.utils import shuffle

from src.logger import logging
from src.exception import CustomException
from src.components.data_cleaning import TextCleaner

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.utils import save_object

# df = pd.read_csv('/Users/kumarshivam/IdeaProjects/New_Job1/data/complaints-2021-05-14_08_16_.json')
a = pd.read_csv('/Users/kumarshivam/IdeaProjects/New_Job1/src/components/artifacts/train.csv', low_memory=False)
b = pd.read_csv('/Users/kumarshivam/IdeaProjects/New_Job1/src/components/artifacts/test.csv')


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_columns = [
                "_source.issue"
            ]

            target_columns = [
                "_source.product"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("Label_Encoder", LabelEncoder())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("data_cleaning", TextCleaner()),
                    ("tfidf_vect", TfidfVectorizer())
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {target_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                    ("num_pipelines", num_pipeline, target_columns)
                ]

            )
            logging.info('Column Transformed')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Reading data")
            a_1 = a[['_source.issue']]
            b_1 = a['_source.product']

            a_2 = b['_source.issue']
            b_2 = b['_source.product']
            logging.info('Reading Done')

            logging.info(a_1)
            logging.info(b_1)

            preprocessing_obj = self.get_data_transformer_object()

            logging.info('Preprocessing Started')
            input_feature_train_arr = preprocessing_obj.fit_transform(a_1)
            input_feature_test_arr = preprocessing_obj.fit_transform(b_1)

            input_target_train_arr = preprocessing_obj.fit_transform(a_2)
            input_target_test_arr = preprocessing_obj.fit_transform(b_2)
            logging.info('Preprocessing Done')

            b_1 = pd.DataFrame(b_1,columns=['_source.product'])
            b_2 = pd.DataFrame(b_2,columns=['_source.product'])

            train_arr = pd.concat([a_1,b_1])
            test_arr = pd.concat([a_2,b_2])

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj



            )

            logging.info("Completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )












        except Exception as e:
            raise CustomException(e, sys)


# print(stopwords.words('english'))

if __name__ == '__main__':
    obj = DataTransformation()

    obj.initiate_data_transformation(a, b)
