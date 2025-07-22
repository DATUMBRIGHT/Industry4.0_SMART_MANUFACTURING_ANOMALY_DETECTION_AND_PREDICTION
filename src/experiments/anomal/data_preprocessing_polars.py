
import polars as pl

# Load the dataset
try:
    df = pl.read_csv("/home/ubuntu/upload/smart_manufacturing_data.csv")
    print("Dataset loaded successfully. First 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.schema)
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Check for missing values
    print("\nMissing Values:")
    for col in df.columns:
        print(f"{col}: {df[col].is_null().sum()}")

    # Convert \'timestamp\' column to datetime if it exists
    if "timestamp" in df.columns:
        df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"))
        print("\nTimestamp column converted to datetime.")
        print(df.head())

    # Save the preprocessed data (optional, for later use)
    df.write_csv("/home/ubuntu/preprocessed_smart_manufacturing_data.csv")
    print("\nPreprocessed data saved to preprocessed_smart_manufacturing_data.csv")

except FileNotFoundError:
    print("Error: smart_manufacturing_data.csv not found. Please ensure the file is in the correct path.")
except Exception as e:
    print(f"An error occurred: {e}")


