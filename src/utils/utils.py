import os,sys
from framework.exception import MyException
from framework.logger import logging
import yaml
import pandas as pd
import numpy as np
import pickle,json



def read_yaml(yaml_file_name: str) -> dict:
    try:
        logging.info('Reading schema file {}'.format(yaml_file_name))
        with open(yaml_file_name, 'r') as f:
            schema = yaml.safe_load(f)
            return schema
    except Exception as e:
        logging.error(e)
        raise MyException(e, sys)

def validate_num_cols(schema_config,df: pd.DataFrame) -> bool:
    try:

        num_cols = len(schema_config)
        logging.info(f"Required no. of columns: {num_cols}")
        logging.info(f"columns in dataframe : {len(df.columns)}")

        if num_cols != len(df.columns):
            logging.error(f"Number of columns does not match number of columns in dataframe")
            return False
        else:
            return True
    except Exception as e:
        logging.error(e)
        raise MyException(e, sys)

def validate_is_null(df: pd.DataFrame) -> bool:
    try:
        null_count = df.isnull().sum().sum()
        logging.info(f"Required no. of null values: {null_count}")
        if null_count != 0:
            logging.error(f"Number of null values does not match number of columns in dataframe")
            return False
        else:
            return True
    except Exception as e:
        logging.error(e)
        raise MyException(e, sys)

def validate_is_numeric(schema_config,df: pd.DataFrame) -> bool:
    try:
        numeric_count = df.select_dtypes(include=[np.number]).count().count()
        logging.info(f"Required no. of numeric values: {numeric_count}")
        if numeric_count != len(schema_config):
            logging.error(f"Number of numeric values does not match number of columns in dataframe")
            return False
        else:
            return True
    except Exception as e:
        logging.error(e)
        raise MyException(e, sys)


def save_numpy_array_data(file_path:str,array:np.array):
    '''
    This function saves the numpy array data
    :param file_path:
    :param array:
    :return: None
    '''
    try:
        logging.info('Saving numpy array data')
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'wb') as f:
            np.save(f, array)
        logging.info(f'Saved successfully numpy array data to {file_path}')
    except Exception as e:
        logging.error(e)
        raise MyException(e, sys)

def save_object(file_path:str,obj:object):
    '''
    This function saves the object data
    :param file_path:
    :param obj:
    :return: None
    '''

    try:
        logging.info('Saving object data')
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'wb') as f_obj:
            pickle.dump(obj, f_obj)
        logging.info(f'Saved successfully object data to {file_path}')
    except Exception as e:
        logging.error(e)
        raise MyException(e, sys)

def load_numpy_array_data(file_path:str):
    '''
    This function loads the numpy array data
    :param file_path:
    :return: numpy array
    '''
    try:
        logging.info('Loading numpy array data')
        if not os.path.exists(file_path):
            logging.error(f'File {file_path} does not exist')
            raise FileNotFoundError(f"File {file_path} does not exist")
        else:
            with open(file_path, 'rb') as file:
                array = np.load(file)
        logging.info(f'Loaded successfully numpy array data to {file_path}')
        return array
    except Exception as e:
        logging.error(e)
        raise MyException(e, sys)

def load_object(file_path:str):
    '''
    This function loads the object data
    :param file_path:
    :return: obj
    '''
    try:
        logging.info('Loading object data')
        if not os.path.exists(file_path):
            logging.error(f'File {file_path} does not exist')
            raise FileNotFoundError(f'File {file_path} does not exist')
        else:
            with open(file_path, 'rb') as f_obj:
                obj = pickle.load(f_obj)
        logging.info(f'Loaded successfully object data to {file_path}')
        return obj
    except Exception as e:
        logging.error(e)
        raise MyException(e, sys)

