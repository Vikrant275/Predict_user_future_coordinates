import os,sys
from framework.exception import MyException
from framework.logger import logging
from datetime import datetime
from src.constant import *

# from Predict_user_future_coordinates.src.constant import DATA_VALIDATION_DIR_NAME


class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_name = ARTIFACT_DIR
            self.timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
            self.artifact_dir = os.path.join(self.artifact_name, self.timestamp)
        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)


class DataIngestionConfig:
    def __init__(self,training_pipeline:TrainingPipelineConfig):
        try:
            self.training_pipeline = training_pipeline

            self.ingestion_dir = os.path.join(self.training_pipeline.artifact_dir,DATA_INGESTION_DIR_NAME)
            self.feature_store_dir = os.path.join(self.ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR)
            self.training_data_file_path = os.path.join(self.ingestion_dir,TRAIN_DATA_FILE)
            self.test_data_file_path = os.path.join(self.ingestion_dir,TEST_DATA_FILE)
            self.train_test_split_ratio = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

            self.input_data_path = INPUT_DIR

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

class FeatureStoreConfig:
    def __init__(self,training_pipeline:TrainingPipelineConfig):
        try:
            self.training_pipeline = training_pipeline

            self.feature_eng_dir = os.path.join(self.training_pipeline.artifact_dir, DATA_FEATURE_ENG_DIR_NAME)
            self.feature_eng_file = os.path.join(self.feature_eng_dir, DATA_FEATURE_ENG_FILE)


        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

class DataValidationConfig:
    def __init__(self,training_pipeline:TrainingPipelineConfig):
        try:
            self.training_pipeline = training_pipeline

            self.data_validation_dir = os.path.join(self.training_pipeline.artifact_dir,DATA_VALIDATION_DIR_NAME)
            self.training_data_file_name = os.path.join(self.data_validation_dir,TRAINING_DATA_FILE_NAME)
            self.testing_data_file_name = os.path.join(self.data_validation_dir,TEST_DATA_FILE_NAME)

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)



