
import pandas as pd

# Load the dataset
try:
    df = pd.read_csv('/home/ubuntu/upload/smart_manufacturing_data.csv')
    print('Dataset loaded successfully. First 5 rows:')
    print(df.head().to_markdown(index=False, numalign="left", stralign="left"))
    print('\nDataset Info:')
    df.info()
    print('\nDescriptive Statistics:')
    print(df.describe().to_markdown(numalign="left", stralign="left"))

    # Check for missing values
    print('\nMissing Values:')
    print(df.isnull().sum().to_markdown(numalign="left", stralign="left"))

    # Convert 'timestamp' column to datetime if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print('\nTimestamp column converted to datetime.')
        print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

    # Save the preprocessed data (optional, for later use)
    df.to_csv('/home/ubuntu/preprocessed_smart_manufacturing_data.csv', index=False)
    print('\nPreprocessed data saved to preprocessed_smart_manufacturing_data.csv')

except FileNotFoundError:
    print('Error: smart_manufacturing_data.csv not found. Please ensure the file is in the correct path.')
except Exception as e:
    print(f'An error occurred: {e}')


