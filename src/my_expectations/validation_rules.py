import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import BaseDataContext
import polars as pl
import pandas as pd
from typing import List

class ManufacturingDataValidator:
    def __init__(self, data_context: BaseDataContext):
        self.data_context = ge.get_context()
        self.data_context.add_datasource("iot_data", "file_path", "data/processed/iot_data.csv")
        self.data_context.add_expectation_suite("iot_data_suite")
        self.suite = self.data_context.get_expectation_suite("iot_data_suite")
        self.validator = self.get_validator()

    def get_validator(self):
        validator = self.data_context.get_validator(
            batch_request=RuntimeBatchRequest(
                datasource_name="iot_data",
                data_connector_name="default_data_connector",
                data_asset_name="iot_data",
            )
        )
        return validator
    
    def validate_data(self, df: pl.DataFrame):
        # --- Rules for machine_id ---
        self.validator.expect_column_values_to_not_be_null("machine_id")
        self.validator.expect_column_values_to_be_between("machine_id", min_value=1, max_value=50)

        # --- Rules for anomaly_flag ---
        self.validator.expect_column_values_to_not_be_null("anomaly_flag")
        self.validator.expect_column_values_to_be_in_set("anomaly_flag", [0, 1])

        # --- Timestamp ---
        self.validator.expect_column_values_to_not_be_null("timestamp")
        self.validator.expect_column_values_to_be_datetime("timestamp")

        # --- One-hot encoded columns ---
        # Machine ID one-hot
        machine_id_cols = [f"machine_id_{i}" for i in range(1, 51)]
        for col in machine_id_cols:
            self.validator.expect_column_values_to_be_in_set(col, [0, 1])
        # Machine status one-hot
        machine_status_cols = ["machine_status_0", "machine_status_1", "machine_status_2"]
        for col in machine_status_cols:
            self.validator.expect_column_values_to_be_in_set(col, [0, 1])
        # Failure type one-hot
        failure_type_cols = [
            "failure_type_Electrical Fault", "failure_type_Normal", "failure_type_Overheating",
            "failure_type_Pressure Drop", "failure_type_Vibration Issue"
        ]
        for col in failure_type_cols:
            self.validator.expect_column_values_to_be_in_set(col, [0, 1])

        # --- Numeric columns ---
        numeric_cols = [
            "temperature", "vibration", "humidity", "pressure", "energy_consumption",
            "downtime_risk", "maintenance_required", "predicted_remaining_life", "anomaly_flag"
        ]
        for col in numeric_cols:
            self.validator.expect_column_values_to_not_be_null(col)
            self.validator.expect_column_values_to_be_of_type(col, "float")  # or "int" as appropriate

       
    def save_expectation(self):
        """--- Save the suite after adding rules ---"""
        self.validator.save_expectation_suite()

        # --- Run validation and print results ---
        results = self.validator.validate()
        print(results)




  