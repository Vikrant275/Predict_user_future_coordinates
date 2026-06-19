import os,sys
from framework.exception import MyException
from framework.logger import logging
from src.component.data_transformation import DataTransformation

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.artifact_entity import FeatureStoreArtifact
from src.entity.artifact_entity import DataValidationArtifact
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.artifact_entity import ModelTrainingArtifact


from src.entity.config_entity import TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig
from src.entity.config_entity import FeatureStoreConfig
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import ModelTraningConfig



from src.component.data_ingestion import DataIngestion
from src.component.feature_engineering import FeatureEng
from src.component.validation import DataValidation
from src.component.model_trainer import ModelTrainer



class TrainingPipeline:
    def __init__(self):
        self.training_pipeline = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")

            data_ingestion_config = DataIngestionConfig(self.training_pipeline)
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion initiated successfully")

            return data_ingestion_artifact
        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def start_feature_engineering(self,data_ingestion_artifact : DataIngestionArtifact):
        try:
            logging.info("Starting feature engineering")
            feature_eng_config = FeatureStoreConfig(self.training_pipeline)
            feature_eng = FeatureEng(data_ingestion_artifact=data_ingestion_artifact,feature_eng_config=feature_eng_config)

            feature_eng_artifact = feature_eng.initiate_feature_engineering()
            logging.info("Feature engineering initiated successfully")

            return feature_eng_artifact

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def start_data_validation(self,feature_store_artifact : FeatureStoreArtifact):
        try:
            logging.info("Starting data validation")

            data_validation_config = DataValidationConfig(self.training_pipeline)
            data_validation = DataValidation(feature_store_artifact=feature_store_artifact, data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation initiated successfully")
            return data_validation_artifact

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def start_data_transformation(self,data_validation_artifact : DataValidationArtifact):
        try:
            logging.info("Starting data transformation")

            data_transformation_config = DataTransformationConfig(self.training_pipeline)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation initiated successfully")
            return data_transformation_artifact

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def start_model_training(self,data_transformation_artifact : DataTransformationArtifact):
        try:
            logging.info("Starting model training")

            model_training_config = ModelTraningConfig(self.training_pipeline)
            model_training = ModelTrainer(data_transformation_artifact=data_transformation_artifact,model_trainer_config=model_training_config)
            model_training_artifact = model_training.initiate_model_training()
            logging.info("Model training initiated successfully")
            return model_training_artifact

        except  Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def run_pipeline(self):
        try:
            logging.info("Starting pipeline execution")

            data_ingestion_artifact = self.start_data_ingestion()
            feature_store_artifact = self.start_feature_engineering(data_ingestion_artifact)
            data_validation_artifact = self.start_data_validation(feature_store_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_training_artifact = self.start_model_training(data_transformation_artifact)


            logging.info('Training pipeline executed successfully')

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)


if __name__ == '__main__':
    training_pipeline = TrainingPipeline()
    training_pipeline.run_pipeline()

