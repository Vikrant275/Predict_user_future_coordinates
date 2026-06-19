import pandas as pd
import os,sys
from framework.exception import MyException
from framework.logger import logging


from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)


    def export_input_into_data_frame(self) -> pd.DataFrame:
        '''
        This function work as fetch data from MongoDB and export it as dataframe.
        :param self:
        :return: dataframe
        '''
        try:

            input_data_path = self.data_ingestion_config.input_data_path
            logging.info(f'Exporting input data to dataframe from path: {input_data_path}')

            data_frame = pd.read_csv(input_data_path)

            logging.info('Exported input data to dataframe')
            return data_frame
        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def export_dataframe_into_feature_store(self,data: pd.DataFrame) -> None:
        '''
        This function work as fetch data as dataframe and export it as feature store at local path.
        :param dataframe:
        :return: feature_store_file_path dataframe
        '''
        try:
            logging.info('Exporting feature store to dataframe')
            feature_dir = os.path.dirname(self.data_ingestion_config.feature_store_dir)
            os.makedirs(feature_dir, exist_ok=True)
            logging.info(f'successfully created feature store {self.data_ingestion_config.feature_store_dir}')

            data.to_csv(self.data_ingestion_config.feature_store_dir,index=False,header=True)

            logging.info('Exported feature store to dataframe')
        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def initiate_data_ingestion(self):
        '''
        This function initiate data ingestion process.
        :return:
        '''

        try:
            logging.info('Initiating data ingestion')

            data_frame = self.export_input_into_data_frame()
            self.export_dataframe_into_feature_store(data_frame)

            data_ingestion_artifact = DataIngestionArtifact(
                feature_store=self.data_ingestion_config.feature_store_dir
            )

            return data_ingestion_artifact

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)


