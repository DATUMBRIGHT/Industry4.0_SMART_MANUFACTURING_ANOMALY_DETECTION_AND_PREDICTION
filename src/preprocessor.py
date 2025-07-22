from typing import Dict, Any
import polars as pl
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import numpy as np 
import os



class Preprocessor:

    def __init__(self,source_file:str,output_file:str):
        self.scaler = StandardScaler()
        self.log_scaler = FunctionTransformer(func=np.log1p, validate=True)
        self.source_file = source_file
        self.output_file = output_file
        self.df = None


    def preprocess_data(self):
        
        """ fucnction to preprocess colun=mns 
         timestamp' to datetime
         vibration,temperature,pressure,humidity,energy_consumption,predicted_remaining_life to z-score
         machine_status to one hot encoding
         failure_type to one hot encoding
         downtime_risk to z-score
         maintenance_required to one hot encoding
         """
       


        #check for null or nan values 
        try:
           df = pl.read_csv(self.source_file)        
           print('file read successfully')
        
        except Exception as e:
            print(f"Error reading or cleaning file: {e}")
            return None
            
        try:
            df = df.drop_nulls()
            print('null values dropped')
        except Exception as e:
            print(f"Error dropping null values: {e}")
            return None
        #convert machine_id to one hot encoding
        if "machine_id" in df.columns:
            try:
                df = df.to_dummies(columns=["machine_id"])
                print("machine_id one-hot encoded")
            except Exception as e:
                print(f"Error encoding machine_id: {e}")
        # Convert timestamp to datetime
        if "timestamp" in df.columns:
            try:
                df = df.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
                )
                print('Timestamp converted to datetime')
            except Exception as e:
                print(f"Error converting timestamp: {e}")

        # Numeric columns to scale
        zscore_cols = ["temperature", "vibration", "pressure", "predicted_remaining_life", "downtime_risk"]
        for col in zscore_cols:
            if col in df.columns:
                try:
                    arr = df[col].to_numpy().reshape(-1, 1)
                    arr_scaled = self.scaler.fit_transform(arr).flatten()
                    df = df.with_columns(pl.Series(col, arr_scaled))
                    print(f"{col} scaled (z-score)")
                except Exception as e:
                    print(f"Error scaling {col}: {e}")

        # Humidity: min-max scaling (30-80)
        if "humidity" in df.columns:
            try:
                arr = df["humidity"].to_numpy()
                arr_scaled = (arr - 30.0) / 50.0
                df = df.with_columns(pl.Series("humidity", arr_scaled))
                print("humidity scaled (min-max)")
            except Exception as e:
                print(f"Error scaling humidity: {e}")

        # Energy consumption: log + z-score
        if "energy_consumption" in df.columns:
            try:
                arr = df["energy_consumption"].to_numpy().reshape(-1, 1)
                arr_log = self.log_scaler.fit_transform(arr)
                arr_scaled = self.scaler.fit_transform(arr_log).flatten()
                df = df.with_columns(pl.Series("energy_consumption", arr_scaled))
                print("energy_consumption scaled (log + z-score)")
            except Exception as e:
                print(f"Error scaling energy_consumption: {e}")

        # One-hot encoding for categorical columns
        if "machine_status" in df.columns:
            try:
                df = df.to_dummies(columns=["machine_status"])
                print("machine_status one-hot encoded")
            except Exception as e:
                print(f"Error encoding machine_status: {e}")

        if "failure_type" in df.columns:
            try:
                df = df.to_dummies(columns=["failure_type"])
                print("failure_type one-hot encoded")
            except Exception as e:
                print(f"Error encoding failure_type: {e}")

        # No scaling for binary columns (anomaly_flag, maintenance_required)
        self.df = df
        return df
    

    def save_preprocessed_data(self):
            try:
                self.df.write_csv(self.output_file)
                print(f"Preprocessed data saved to {self.output_file}")
            except Exception as e:
                print(f"Error saving preprocessed data: {e}")
                return self.df


