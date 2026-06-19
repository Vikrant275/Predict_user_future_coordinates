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