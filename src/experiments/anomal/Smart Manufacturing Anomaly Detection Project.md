# Smart Manufacturing Anomaly Detection Project

This project implements advanced anomaly detection for smart manufacturing data using PatchTST and Temporal Fusion Transformer (TFT) models.

## Files Included

1. **anomaly_detection_notebook_content.ipynb** - Main Jupyter Notebook with complete analysis
2. **preprocessed_smart_manufacturing_data.csv** - Preprocessed dataset ready for analysis
3. **data_preprocessing_polars.py** - Data preprocessing script using Polars
4. **patchtst_anomaly_detection.py** - Standalone PatchTST implementation
5. **tft_anomaly_detection.py** - Standalone TFT implementation

## Requirements

Install the following Python packages:

```bash
pip install polars numpy scikit-learn torch matplotlib seaborn jupytext
```

## Running the Analysis

### Option 1: Jupyter Notebook (Recommended)
1. Open `anomaly_detection_notebook_content.ipynb` in Jupyter Lab or Jupyter Notebook
2. Run all cells sequentially
3. The notebook includes detailed explanations, visualizations, and insights

### Option 2: Standalone Scripts
1. Run data preprocessing: `python data_preprocessing_polars.py`
2. Run PatchTST analysis: `python patchtst_anomaly_detection.py`
3. Run TFT analysis: `python tft_anomaly_detection.py`

## Key Features

### PatchTST Model
- Uses patch-based transformer architecture
- Detects anomalies through reconstruction error
- Effective for capturing local temporal patterns

### TFT Model
- Implements simplified Temporal Fusion Transformer
- Detects anomalies through prediction error
- Leverages multi-variate time series forecasting

### Data Processing
- Utilizes Polars for efficient data manipulation
- Handles 100,000 data points across 13 features
- Includes timestamp conversion and missing value analysis

## Business Insights

The analysis provides:
- Anomaly detection on temperature, vibration, humidity, pressure, and energy consumption
- Comparative analysis between PatchTST and TFT approaches
- Visualization of detected anomalies
- Actionable recommendations for manufacturing optimization

## Recommendations

1. **Ensemble Approach**: Combine both models for robust anomaly detection
2. **Threshold Tuning**: Adjust anomaly thresholds based on domain expertise
3. **Real-time Monitoring**: Implement continuous monitoring system
4. **Root Cause Analysis**: Investigate persistent anomalies for process improvement

## Model Performance Notes

- Models are simplified implementations suitable for demonstration
- For production use, consider full implementations with hyperparameter tuning
- Training time depends on available computational resources
- Results may vary based on data characteristics and model parameters

## Support

For questions or issues, refer to the detailed comments and documentation within the Jupyter Notebook.

