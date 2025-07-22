from preprocessor import Preprocessor
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from my_expectations.validation_rules import ManufacturingDataValidator

if __name__ == "__main__":
    """preprocessor = Preprocessor(source_file="/Users/brighttenkorangofori/Desktop/projects/industry4.0/src/data/raw/smart_manufacturing_data.csv", output_file="/Users/brighttenkorangofori/Desktop/projects/industry4.0/src/data/processed/iot_data.csv")
    os.makedirs(os.path.dirname(preprocessor.output_file), exist_ok=True)
    preprocessor.preprocess_data()
    df =  preprocessor.save_preprocessed_data()
    print(df.head(3))"""



    #validate the data
    validator = ManufacturingDataValidator()
    validator.validate_data()
    validator.save_expectation()