from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store : str

@dataclass
class FeatureStoreArtifact:
    feature_eng : str

@dataclass
class DataValidationArtifact:
    training_data : str
    testing_data : str

@dataclass
class DataTransformationArtifact:
    training_data : str
    testing_data : str
    preprocessor : str

@dataclass
class ClassificationMetricsArtifact:
    f1_score : float
    recall_score : float
    precision_score : float

@dataclass()
class ModelTrainingArtifact:
    model_name:str
    trained_model_file_path : str
    train_metrics_artifact : ClassificationMetricsArtifact
    test_metrics_artifact : ClassificationMetricsArtifact
