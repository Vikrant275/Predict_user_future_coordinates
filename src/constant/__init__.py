import os,sys
from framework.fetch_config import GetConfig

'''
Constants for training pipeline
'''

ARTIFACT_DIR = GetConfig(config_file='dir_path.yaml',variables='artifact').get()
INPUT_DIR = GetConfig(config_file='file_name.yaml',variables='input_file').get()

TRAIN_DATA_FILE = GetConfig(config_file='file_name.yaml',variables='training_data').get()
TEST_DATA_FILE = GetConfig(config_file='file_name.yaml',variables='test_data').get()


'''
constants for data ingestion 
'''

DATA_INGESTION_DIR_NAME : str= 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR : str= 'feature_store.csv'
DATAINGESTION_INGESTED_DIR : str= 'ingested_data'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO : float= 0.25


'''
constants for feature engineering
'''
DATA_FEATURE_ENG_DIR_NAME : str= 'feature_eng'
DATA_FEATURE_ENG_FILE : str= 'feature_engineering.csv'

'''
Validation constants
'''
DATA_VALIDATION_DIR_NAME : str= 'validation'
TRAINING_DATA_FILE_NAME : str= 'training_data.csv'
TEST_DATA_FILE_NAME : str= 'test_data.csv'
SCHEMA_FILE_NAME : str = 'D:\\Pertsol\\Predict_user_future_coordinates\\schema\\schema.yaml'