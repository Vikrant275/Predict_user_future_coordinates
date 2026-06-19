import os,sys

# from sklearn.externals.array_api_compat.torch import result_type

from framework.exception import MyException
from framework.logger import logging

import mlflow
from src.entity.config_entity import ModelTraningConfig
from src.entity.artifact_entity import ClassificationMetricsArtifact,ModelTrainingArtifact,DataTransformationArtifact

from src.utils.utils import *

from src.utils.ml_utils import *

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor


class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTraningConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    # def track_mlflow(self,model,train_classification,test_classification):
    #     with mlflow.start_run() as run:
    #
    #         mlflow.sklearn.log_model(
    #             sk_model=model,
    #             artifact_path="model",
    #             skops_trusted_types=[
    #                 "xgboost.core.Booster",
    #                 "xgboost.sklearn.XGBClassifier"
    #             ]
    #         )
    #
    #         mlflow.log_param("train_f1_score",train_classification.f1_score)
    #         mlflow.log_param("train_precision",train_classification.precision_score)
    #         mlflow.log_param("train_recall",train_classification.recall_score)
    #
    #         mlflow.log_param("test_f1_score", test_classification.f1_score)
    #         mlflow.log_param("test_precision", test_classification.precision_score)
    #         mlflow.log_param("test_recall", test_classification.recall_score)
    #     mlflow.end_run()




    def train_model(self,x_train:np.ndarray,y_train:np.ndarray,x_test:np.ndarray,y_test:np.ndarray):
        models = {
            'RandomForestClassifier':RandomForestClassifier(),
            'GradientBoostingClassifier':GradientBoostingClassifier(),
            'XGBClassifier':XGBClassifier()
        }

        param_grids = {

            'RandomForestClassifier': {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 20, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            'GradientBoostingClassifier': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 10],
                'subsample': [0.8, 1.0],
                'min_samples_split': [2, 5, 10]
            },

            'XGBClassifier': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 10],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [1, 1.5, 2]
            }
        }

        model_report:dict = EvaluateModel(x_train,y_train,x_test,y_test,models,param_grids).evaluate_model()
        best_entry = max(
            model_report.items(),
            key=lambda x: (x[1]['f1_score_test'] + x[1]['precision_test'] + x[1]['recall_test']) / 3
        )
        logging.info(f"{best_entry} ------------")
        model = best_entry[1]['best_model']
        model_name = best_entry[0]
        logging.info(f"model evaluation successfully")
        logging.info(f"best model: {model}  with best score: {best_entry[1]}")

        y_train_pred = model.predict(x_train)
        classification_train_metric = ClassificationMetrics(y_train,y_train_pred).get_classification_metrics()

        y_test_pred = model.predict(x_test)
        classification_test_metric = ClassificationMetrics(y_test,y_test_pred).get_classification_metrics()

        # # Track mlflow
        # logging.info("mlflow tracking started")
        #
        # tracking_dir = os.path.abspath("D:\\Pertsol\\Predict_user_future_coordinates\\mlruns")
        # mlflow.set_tracking_uri("file:\\" + tracking_dir)
        #
        # self.track_mlflow(model ,classification_train_metric,classification_test_metric)
        # logging.info("mlflow tracking ended")

        logging.info(f"loading preprocessed pkl file")
        preprocessor  = load_object(file_path=self.data_transformation_artifact.preprocessor)

        model_dir_path = os.path.join(self.model_trainer_config.model_training_dir)
        os.makedirs(model_dir_path, exist_ok=True)
        model_path = os.path.join(model_dir_path, self.model_trainer_config.trained_model_name)
        logging.info(f"created model directory at {model_dir_path}")


        future_model = FutureLocationModel(preprocessor=preprocessor,model=model)
        save_object(file_path=model_path,obj=future_model)
        save_object(file_path='final_model/model.pkl',obj=model)

        model_train_artifact = ModelTrainingArtifact(
            model_name=model_name,
            trained_model_file_path=model_path,
            train_metrics_artifact=classification_train_metric,
            test_metrics_artifact=classification_test_metric

        )

        return model_train_artifact


    def initiate_model_training(self) -> ModelTrainingArtifact:
        try:
            logging.info("Initiating model training")

            train_file_path = self.data_transformation_artifact.training_data
            test_file_path = self.data_transformation_artifact.testing_data

            logging.info("loading training data from {}".format(train_file_path))
            train_array = load_numpy_array_data(train_file_path)

            logging.info("loading test data from {}".format(test_file_path))
            test_array = load_numpy_array_data(test_file_path)

            logging.info(f"splitting training  and testing data into input feature and output feature ")

            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model_artifact = self.train_model(x_train,y_train,x_test,y_test)
            return model_artifact

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)
