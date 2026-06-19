from src.entity.artifact_entity import ClassificationMetricsArtifact

from framework.logger import logging
from framework.exception import MyException,sys

from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.model_selection import RandomizedSearchCV


class ClassificationMetrics:
    def __init__(self,y_true,y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def get_classification_metrics(self) -> ClassificationMetricsArtifact:
        try:
            model_f1_score = f1_score(self.y_true, self.y_pred, average='micro')
            model_recall_score = recall_score(self.y_true, self.y_pred, average='micro')
            model_precision_score = precision_score(self.y_true, self.y_pred, average='micro')
            calssification_metric = ClassificationMetricsArtifact(
                f1_score = model_f1_score,
                recall_score = model_recall_score,
                precision_score = model_precision_score,
            )
            return calssification_metric
        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)


class EvaluateModel:
    def __init__(self,x_train,y_train,x_test,y_test,models,param_grids):
        self.models = models
        self.param_grids = param_grids
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    def evaluate_model(self):
        try:
            report={}
            logging.info("Evaluating model....")

            for name,model in self.models.items():
                random = RandomizedSearchCV(estimator=model,param_distributions=self.param_grids[name],n_iter=10,cv=5,scoring='accuracy',verbose=1,n_jobs=-1)
                random.fit(self.x_train,self.y_train)

                y_pred = random.predict(self.x_test)

                best_model = random.best_estimator_

                classification_matrics = ClassificationMetrics(self.y_test, y_pred).get_classification_metrics()
                f1_score_test= classification_matrics.f1_score
                precision_test= classification_matrics.precision_score
                recall_test= classification_matrics.recall_score

                report.update({
                    name:{
                        'best_model':best_model,
                        'f1_score_test':f1_score_test,
                        'precision_test':precision_test,
                        'recall_test':recall_test,
                        }
                    })
            return report

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)


class FutureLocationModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)



