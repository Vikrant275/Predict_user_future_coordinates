import pandas as pd
import numpy as np
import os,sys
from framework.exception import MyException
from framework.logger import logging

from sklearn.cluster import DBSCAN


from math import radians
from math import sin
from math import cos
from math import sqrt
from math import atan2

from src.entity.config_entity import FeatureStoreConfig
from src.entity.artifact_entity import FeatureStoreArtifact
from src.entity.artifact_entity import DataIngestionArtifact


class FeatureEng:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,
                 feature_eng_config:FeatureStoreConfig):
        try:
            self.feature_eng_config = feature_eng_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logging.error(e)
            raise MyException(e, sys)

    def create_time_feature(self,gps_df : pd.DataFrame) -> pd.DataFrame:
        try:

            gps_df["timestamp"] = pd.to_datetime(
                gps_df["timestamp"],
                format="%Y-%m-%d %H:%M"
            )

            gps_df["hour"] = gps_df["timestamp"].dt.hour
            gps_df["day_of_week"] = gps_df["timestamp"].dt.dayofweek
            gps_df["month"] = gps_df["timestamp"].dt.month

            gps_df["is_weekend"] = (
                    gps_df["day_of_week"] >= 5
            ).astype(int)

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)
        return gps_df
    def create_cyclical_feature(self,gps_df : pd.DataFrame) -> pd.DataFrame:

        try:

            gps_df["hour_sin"] = np.sin(
                2 * np.pi * gps_df["hour"] / 24
            )

            gps_df["hour_cos"] = np.cos(
                2 * np.pi * gps_df["hour"] / 24
            )

            return gps_df

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def cluster_features(self,gps_df : pd.DataFrame) -> pd.DataFrame:
        try:

            coords = gps_df[
                ["latitude", "longitude"]
            ]

            dbscan = DBSCAN(
                eps=0.003,
                min_samples=10
            )

            gps_df["location_id"] = dbscan.fit_predict(coords)

            return gps_df

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def create_previous_location_feature(self,gps_df : pd.DataFrame) -> pd.DataFrame:
        try:

            gps_df["prev_location_1"] = (
                gps_df["location_id"].shift(1)
            )

            gps_df["prev_location_2"] = (
                gps_df["location_id"].shift(2)
            )

            gps_df["prev_location_3"] = (
                gps_df["location_id"].shift(3)
            )

            return gps_df

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def create_distance_travelled_feature(self, gps_df : pd.DataFrame) -> pd.DataFrame:
        try:

            '''The Haversine formula calculates the shortest distance 
                between two points on the surface of a sphere (the Earth) given their longitudes and latitudes
            '''

            def haversine(lat1, lon1, lat2, lon2):

                R = 6371  # radius of earth

                '''Trigonometric functions in Python (sin, cos) 
                   require angles to be in radians, not degrees. This converts your GPS coordinates from degrees to radians
                '''
                dlat = radians(lat2 - lat1)
                dlon = radians(lon2 - lon1)

                ''' This calculates the square of half the chord length between the two points
                '''
                a = (
                        sin(dlat / 2) ** 2
                        +
                        cos(radians(lat1))
                        *
                        cos(radians(lat2))
                        *
                        sin(dlon / 2) ** 2
                )

                '''This calculates the angular distance in radians using 
                    atan2 (arcsine variant), which is highly stable against rounding errors
                '''

                c = 2 * atan2(
                    sqrt(a),
                    sqrt(1 - a)
                )

                ''' Multiplying the angular distance by the Earth's radius yields the final distance in kilometres.
                '''
                return R * c

            gps_df["prev_lat"] = gps_df["latitude"].shift(1)
            gps_df["prev_lon"] = gps_df["longitude"].shift(1)

            gps_df["distance_from_prev_km"] = gps_df.apply(
                lambda x: haversine(
                    x["prev_lat"],
                    x["prev_lon"],
                    x["latitude"],
                    x["longitude"]
                ) if pd.notnull(x["prev_lat"]) else 0,
                axis=1
            )

            '''
            Detect Travel / Anomaly

            Large jumps indicate travel.
            '''

            gps_df["anomaly_flag"] = (
                    gps_df["distance_from_prev_km"] > 30
            ).astype(int)

            '''
            Rolling Travel Radius
            '''

            gps_df["travel_radius_24h"] = (
                gps_df["distance_from_prev_km"]
                .rolling(24)
                .sum()
            )

            return gps_df
        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)

    def create_traget_feature(self,gps_df : pd.DataFrame) -> pd.DataFrame:
        try:

            gps_df["target_next_location"] = (
                gps_df["location_id"].shift(-1)
            )

            gps_df.dropna(inplace=True)

            return gps_df
        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)


    def initiate_feature_engineering(self) :
        try:
            logging.info("Initiating feature engineering")

            data = self.read_data(self.data_ingestion_artifact.feature_store)
            logging.info("Feature Engineering data has been read")

            logging.info("Creating Time Feature")
            gps_df = self.create_time_feature(data)
            logging.info("Created Successfully Time Feature")

            logging.info("Creating Cyclical Feature")
            gps_df = self.create_cyclical_feature(gps_df)
            logging.info("Created Successfully Cyclical Feature")

            logging.info("Creating Cluster Feature")
            gps_df= self.cluster_features(gps_df)
            logging.info("Created Successfully Cluster Feature")

            logging.info("Creating Previous Location Feature")
            gps_df = self.create_previous_location_feature(gps_df)
            logging.info("Created Successfully Previous Location Feature")

            logging.info("Creating Distance Feature")
            gps_df = self.create_distance_travelled_feature(gps_df)
            logging.info("Created Successfully Distance Feature")

            logging.info("Creating Traget Feature")
            gps_df = self.create_traget_feature(gps_df)
            logging.info("Created Successfully Traget Feature")

            feature_eng = os.path.dirname(self.feature_eng_config.feature_eng_file)
            os.makedirs(feature_eng, exist_ok=True)

            gps_df.to_csv(self.feature_eng_config.feature_eng_file, index=False,header=True)


            feature_eng_artifact = FeatureStoreArtifact(
                feature_eng= self.feature_eng_config.feature_eng_file
            )


            return feature_eng_artifact

        except Exception as e:
            logging.error(e)
            raise MyException(e,sys)
