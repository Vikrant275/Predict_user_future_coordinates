import pandas as pd
import numpy as np
import os,sys
from framework.exception import MyException
from framework.logger import logging
from typing import Tuple,Dict
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.constant import *
from src.entity.config_entity import DataTransformationConfig

from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.artifact_entity import DataValidationArtifact

from src.utils.utils import *


class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):

        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    @staticmethod
    def read_data(train_file_path,test_file_path) -> Tuple[pd.DataFrame,pd.DataFrame]:
        try:
            return pd.read_csv(train_file_path),pd.read_csv(test_file_path)
        except Exception as e:
            logging.error(e)
            raise MyException(e, sys)

    def apply_standard_scaling(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray,ColumnTransformer]:
        try:

            x_train = train.drop(columns=DROP_COLUMN_NAME)
            y_train = train[TARGET_COLUMNS_NAME]

            x_test = test.drop(columns=DROP_COLUMN_NAME)
            y_test = test[TARGET_COLUMNS_NAME]

            num_col = x_train.select_dtypes(exclude=['object']).columns

            preprocessor = ColumnTransformer([
                ('Scaler', StandardScaler(), num_col)
            ], remainder='passthrough')

            x_train_scaled = preprocessor.fit_transform(x_train)
            x_test_scaled = preprocessor.transform(x_test)

            train_arr = np.c_[x_train_scaled, y_train]
            test_arr = np.c_[x_test_scaled, y_test]

            return train_arr, test_arr, preprocessor

        except Exception as e:
            logging.error(e)
            raise MyException(e, sys)

    def initiate_data_transformation(self):
        try:
            logging.info('reading training data and testing data')
            train_df,test_df = self.read_data(self.data_validation_artifact.training_data, self.data_validation_artifact.testing_data)

            train_arr,test_arr,preprocessor = self.apply_standard_scaling(train_df,test_df)

            logging.info('successfully apply data transformation on train and test data')

            data_transformation_dir_name = os.path.dirname(self.data_transformation_config.data_transformation_dir)
            os.makedirs(self.data_transformation_config.data_transformation_dir,exist_ok=True)

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            logging.info(
                f'Saved scaled train array at {[self.data_transformation_config.transformed_train_file_path]} with shape {train_arr.shape}')

            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            logging.info(
                f'Saved scaled test array at {[self.data_transformation_config.transformed_test_file_path]} with shape {test_arr.shape}')

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            save_object(file_path='final_preprocessor/preprocessor.pkl', obj=preprocessor)

            data_transformation_artifact = DataTransformationArtifact(
                training_data=self.data_transformation_config.transformed_train_file_path,
                testing_data=self.data_transformation_config.transformed_test_file_path,
                preprocessor=self.data_transformation_config.transformed_object_file_path

            )


            return data_transformation_artifact

        except Exception as e:
                logging.error(e)
                raise MyException(e,sys)
