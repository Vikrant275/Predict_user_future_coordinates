import os,sys

from sklearn.utils import TargetTags

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
TRAINING_DATA_FILE_NAME : str= GetConfig(config_file='file_name.yaml',variables='training_data').get()
TEST_DATA_FILE_NAME : str= GetConfig(config_file='file_name.yaml',variables='test_data').get()
SCHEMA_FILE_NAME : str = 'D:\\Pertsol\\Predict_user_future_coordinates\\schema\\schema.yaml'


'''
constants for data transformation
'''

DATA_TRANSFORMATION_DIR_NAME :str = 'transformation_dir'
DATA_TRANSFORMATION_TRAIN_FILE :str = GetConfig(config_file='file_name.yaml',variables='transformed_train').get()
DATA_TRANSFORMATION_TEST_FILE :str = GetConfig(config_file='file_name.yaml',variables='transformed_test').get()
DATA_TRANSFORMATION_OBJECT_FILE :str = GetConfig(config_file='file_name.yaml',variables='transformed_obj').get()
DROP_COLUMN_NAME = ['target_next_location', 'phone_number', 'timestamp']
TARGET_COLUMNS_NAME = 'target_next_location'


'''
Model training related variables
'''
MODEL_TRAINING_DIR_NAME :str = GetConfig(config_file='file_name.yaml', variables='model_train').get()
MODEL_TRAINING_MODEL_NAME : str = GetConfig(config_file='file_name.yaml', variables='model_name').get()
MODEL_TRAINED_EXPECTED_SCORE :float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD : float = 0.05

'''
prediction constants
'''
MODEL_INFORMATION:str = GetConfig(config_file='file_name.yaml', variables='model_information').get()