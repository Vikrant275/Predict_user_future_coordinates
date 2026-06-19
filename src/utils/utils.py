import os,sys
from framework.exception import MyException
from framework.logger import logging
import yaml
import pandas as pd
import numpy as np



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