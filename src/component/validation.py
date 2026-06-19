import pandas as pd
import numpy as np
import os,sys
from framework.exception import MyException
from framework.logger import logging
from typing import Tuple,Dict
from src.constant import *
from src.entity.config_entity import DataValidationConfig

from src.entity.artifact_entity import DataValidationArtifact
from src.entity.artifact_entity import FeatureStoreArtifact

from src.utils.utils import *


class DataValidation:
    def __init__(self,feature_store_artifact: FeatureStoreArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.feature_store_artifact = feature_store_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml(SCHEMA_FILE_NAME)
        except MyException as e:
            logging.error(e)
            raise MyException(e,sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logging.error(e)
            raise MyException(e, sys)

    def validate_data(self,data : pd.DataFrame) -> Dict :
        try:
            status_report = {}
            # Validate no. of columns
            status = validate_num_cols(self._schema_config['columns'], data)
            if not status:
                error_message = f"Number of columns does not match number of columns in dataframe"
                status_report.update({'validate_num_col': {'data': {'status' : status, 'message': error_message}}})

            # validate null
            status = validate_is_null(data)
            if not status:
                error_message = f"Number of null values does not match number of columns in dataframe"
                status_report.update({'validate_null_col': {'data': {'status' : status, 'message': error_message}}})

            # Validate numeric columns
            status = validate_is_numeric(self._schema_config['numerical_columns'], data)
            if not status:
                error_message = f"Number of numeric values does not match number of columns in dataframe"
                status_report.update({'validate_numeric_col': {'data': {'status' : status, 'message': error_message}}})

            return status_report

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def split_data_into_train_test(self, data: pd.DataFrame,test_size=0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            """
                Performs an out-of-time validation split per device.
                Trains on past logs, validates on future logs to prevent data leakage.
                """
            train_slices = []
            test_slices = []

            # Group by device to ensure temporal split integrity per phone
            for _, device_group in data.groupby("phone_number"):
                # Sort to guarantee order
                device_group = device_group.sort_values("timestamp")

                split_idx = int(len(device_group) * (1 - test_size))
                if split_idx == 0 and len(device_group) > 0:
                    split_idx = 1

                train_slices.append(device_group.iloc[:split_idx])
                test_slices.append(device_group.iloc[split_idx:])

            train_df = pd.concat(train_slices).reset_index(drop=True)
            test_df = pd.concat(test_slices).reset_index(drop=True)

            return train_df, test_df

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def initiate_data_validation(self):
        try:
            logging.info("Initiating data validation")

            data = self.read_data(self.feature_store_artifact.feature_eng)
            get_status_report = self.validate_data(data)

            # if (get_status_report['validate_null_col']['data']['status'] != False) or (get_status_report['validate_numeric_col']['data']['status'] != False) or (get_status_report['validate_num_col']['data']['status'] != False) :

            logging.info("Spliting data into train and test")
            train_df, test_df = self.split_data_into_train_test(data)

            logging.info(f'Saving train file at location {[self.data_validation_config.training_data_file_name]}')
            train_dir_name = os.path.dirname(self.data_validation_config.training_data_file_name)
            os.makedirs(train_dir_name, exist_ok=True)

            train_df.to_csv(self.data_validation_config.training_data_file_name,index=False,header=True)
            logging.info(f'Saved train file successully  at location {[self.data_validation_config.training_data_file_name]}')

            logging.info(f'Saving test file at location {[self.data_validation_config.testing_data_file_name]}')
            test_dir_name = os.path.dirname(self.data_validation_config.testing_data_file_name)
            os.makedirs(test_dir_name, exist_ok=True)

            test_df.to_csv(self.data_validation_config.testing_data_file_name,index=False,header=True)
            logging.info(f'Saved test file successully  at location {[self.data_validation_config.testing_data_file_name]}')

            data_validation_artifact = DataValidationArtifact(
                training_data= self.data_validation_config.training_data_file_name,
                testing_data= self.data_validation_config.testing_data_file_name
            )

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

