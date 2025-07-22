
# This script generates the content for a Jupyter Notebook (.ipynb) file.
# You can copy and paste this content into a new .ipynb file or use a tool
# like `jupytext` to convert it to a notebook.

# --- Section 1: Introduction and Setup ---

# # Smart Manufacturing Anomaly Detection

# This notebook performs advanced analytics for anomaly detection on smart manufacturing data.
# We will utilize two powerful time series models: PatchTST and a simplified Temporal Fusion Transformer (TFT).
# The analysis will include data loading, preprocessing, model implementation, anomaly detection, and interpretation of results.

# ## Setup and Library Installation

# First, ensure you have the necessary libraries installed. You can install them using pip:
# ```bash
# pip install polars numpy scikit-learn torch
# ```

# ## Data Loading and Initial Exploration (using Polars)

# We will load the `smart_manufacturing_data.csv` dataset and perform initial data exploration and preprocessing using Polars for efficient data handling.

import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

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
    print(f"An error occurred during data loading: {e}")


# --- Section 2: PatchTST for Anomaly Detection ---

# # PatchTST Model for Anomaly Detection

# PatchTST is a powerful transformer-based model for time series analysis. For anomaly detection, we typically train the model on normal data to learn its patterns. Anomalies are then identified as deviations from these learned patterns, often by measuring reconstruction error.

# ## Model Configuration and Data Preparation

# Define model parameters and prepare the data for PatchTST. We will use \'temperature\' as the primary feature for this example, but the approach can be extended to multivariate data.

SEQUENCE_LENGTH = 96  # Length of input sequences for the model
PREDICTION_LENGTH = 24 # Not directly used for AD, but common in forecasting models
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# --- PatchTST Model Definition (Simplified for Anomaly Detection) ---
# This is a simplified representation. A full PatchTST implementation is complex.
# For actual anomaly detection, you\'d typically use the reconstruction error or
# prediction error from a forecasting task.

class PatchTST(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim

        self.patch_embedding = nn.Linear(patch_len, hidden_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.output_layer = nn.Linear(hidden_dim, patch_len)

    def forward(self, x):
        if self.input_dim > 1:
            raise NotImplementedError("Simplified PatchTST only supports input_dim=1 for patching example.")

        x_univariate = x.squeeze(-1)

        patches = []
        for i in range(0, x_univariate.shape[1] - self.patch_len + 1, self.stride):
            patches.append(x_univariate[:, i : i + self.patch_len])
        
        patches = torch.stack(patches, dim=1)

        embedded_patches = self.patch_embedding(patches)

        encoded_output = self.transformer_encoder(embedded_patches)

        reconstructed_patches = self.output_layer(encoded_output)

        return reconstructed_patches

# --- Data Preparation for PatchTST ---
def prepare_data_patchtst(df, features, sequence_length):
    data = df[features].to_numpy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    X = []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i : i + sequence_length])
    
    X = np.array(X)
    return torch.tensor(X, dtype=torch.float32), scaler

# --- Anomaly Detection Logic for PatchTST ---
def detect_anomalies_patchtst(model, dataloader, threshold_multiplier=3):
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for batch_X in dataloader:
            if isinstance(batch_X, (list, tuple)):
                batch_X = batch_X[0]
            
            reconstructed_patches = model(batch_X)
            
            original_first_patch = batch_X[:, 0, :model.patch_len].unsqueeze(1)
            error = torch.mean((original_first_patch - reconstructed_patches[:, 0, :].unsqueeze(1))**2, dim=[-1, -2])
            reconstruction_errors.extend(error.cpu().numpy())

    errors = np.array(reconstruction_errors)
    threshold = np.mean(errors) + threshold_multiplier * np.std(errors)
    anomalies = errors > threshold
    return anomalies, errors, threshold

# ## PatchTST Execution

# Now, let\'s execute the PatchTST model for anomaly detection.

print("\n--- Running PatchTST Anomaly Detection ---")
try:
    df_patchtst = pl.read_csv("/home/ubuntu/preprocessed_smart_manufacturing_data.csv")
    if "timestamp" in df_patchtst.columns and df_patchtst["timestamp"].dtype != pl.Datetime:
        df_patchtst = df_patchtst.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"))

    features_patchtst = ["temperature"]
    print(f"Preparing data for PatchTST using feature: {features_patchtst[0]}...")
    X_tensor_patchtst, scaler_patchtst = prepare_data_patchtst(df_patchtst, features_patchtst, SEQUENCE_LENGTH)

    dataset_patchtst = TensorDataset(X_tensor_patchtst)
    dataloader_patchtst = DataLoader(dataset_patchtst, batch_size=BATCH_SIZE, shuffle=False)

    input_dim_patchtst = 1
    hidden_dim_patchtst = 64
    num_heads_patchtst = 4
    num_layers_patchtst = 2
    patch_len_patchtst = 16
    stride_patchtst = 8

    print("Initializing PatchTST model...")
    model_patchtst = PatchTST(input_dim_patchtst, hidden_dim_patchtst, num_heads_patchtst, num_layers_patchtst, patch_len_patchtst, stride_patchtst)

    print("Starting simplified PatchTST training loop...")
    optimizer_patchtst = torch.optim.Adam(model_patchtst.parameters(), lr=LEARNING_RATE)
    criterion_patchtst = nn.MSELoss()

    for epoch in range(EPOCHS):
        model_patchtst.train()
        total_loss_patchtst = 0
        for batch_X_patchtst in dataloader_patchtst:
            if isinstance(batch_X_patchtst, (list, tuple)):
                batch_X_patchtst = batch_X_patchtst[0]

            optimizer_patchtst.zero_grad()
            reconstructed_patches_patchtst = model_patchtst(batch_X_patchtst)
            
            original_first_patch_patchtst = batch_X_patchtst[:, 0, :patch_len_patchtst].unsqueeze(1)
            loss_patchtst = criterion_patchtst(reconstructed_patches_patchtst[:, 0, :].unsqueeze(1), original_first_patch_patchtst)
            
            loss_patchtst.backward()
            optimizer_patchtst.step()
            total_loss_patchtst += loss_patchtst.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss_patchtst/len(dataloader_patchtst):.4f}")

    print("Detecting anomalies using PatchTST...")
    anomalies_patchtst, errors_patchtst, threshold_patchtst = detect_anomalies_patchtst(model_patchtst, dataloader_patchtst)

    anomaly_flags_full_patchtst = np.zeros(len(df_patchtst), dtype=bool)
    anomaly_indices_patchtst = np.where(anomalies_patchtst)[0] + SEQUENCE_LENGTH -1
    anomaly_indices_patchtst = anomaly_indices_patchtst[anomaly_indices_patchtst < len(df_patchtst)]
    anomaly_flags_full_patchtst[anomaly_indices_patchtst] = True

    df_patchtst_results = df_patchtst.with_columns(
        pl.Series(name="patchtst_anomaly", values=anomaly_flags_full_patchtst)
    )

    print("\nPatchTST Anomaly Detection Results (first 10 anomalies):")
    print(df_patchtst_results.filter(pl.col("patchtst_anomaly") == True).head(10).to_string())

    df_patchtst_results.write_csv("/home/ubuntu/patchtst_anomaly_results.csv")
    print("PatchTST anomaly results saved to patchtst_anomaly_results.csv")

    # # PatchTST Analysis and Insights

    # The PatchTST model identifies anomalies based on the reconstruction error of time series patches. A higher error indicates a greater deviation from the learned normal patterns.

    # ### Visualizing PatchTST Anomalies

    # Let\'s visualize the detected anomalies on the \'temperature\' data.

    plt.figure(figsize=(15, 6))
    plt.plot(df_patchtst["timestamp"], df_patchtst["temperature"], label="Temperature")
    plt.scatter(
        df_patchtst.filter(pl.col("patchtst_anomaly") == True)["timestamp"],
        df_patchtst.filter(pl.col("patchtst_anomaly") == True)["temperature"],
        color=\'red\', label=\'Anomaly (PatchTST)\', s=50, marker=\'o\'
    )
    plt.title("Temperature with PatchTST Detected Anomalies")
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ### Recommendations for PatchTST

    # *   **Threshold Adjustment**: The current anomaly threshold is based on a simple statistical measure (mean + 3*std). This can be fine-tuned based on domain expertise and desired sensitivity/specificity.
    # *   **Multivariate Analysis**: Extend the PatchTST implementation to handle all relevant numerical features simultaneously for a more comprehensive anomaly detection.
    # *   **Anomaly Scoring**: Instead of just a binary flag, provide an anomaly score (e.g., the reconstruction error itself) to rank anomalies by severity.

except Exception as e:
    print(f"An error occurred during PatchTST execution: {e}")


# --- Section 3: Temporal Fusion Transformer (TFT) for Anomaly Detection ---

# # Temporal Fusion Transformer (TFT) for Anomaly Detection

# The Temporal Fusion Transformer (TFT) is designed for high-performance multi-horizon forecasting. For anomaly detection, we can leverage its forecasting capabilities: if the actual value deviates significantly from the predicted value, it can be flagged as an anomaly.

# ## Model Configuration and Data Preparation

# Define model parameters and prepare the data for TFT. We will use multiple numerical features for this model.

# --- Simplified TFT Model Definition ---
class SimplifiedTFT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded_input = self.input_embedding(x)

        encoded_output = self.transformer_encoder(embedded_input)

        last_hidden_state = encoded_output[:, -1, :]

        predictions = self.decoder(last_hidden_state)

        return predictions

# --- Data Preparation for TFT ---
def prepare_data_tft(df, features, sequence_length, prediction_length):
    data = df[features].to_numpy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - sequence_length - prediction_length + 1):
        X.append(data_scaled[i : i + sequence_length])
        y.append(data_scaled[i + sequence_length : i + sequence_length + prediction_length, 0]) # Predict first feature
    
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

# --- Anomaly Detection Logic for TFT ---
def detect_anomalies_tft(model, dataloader, scaler, original_df, features, threshold_multiplier=3):
    model.eval()
    prediction_errors = []
    original_values = []
    
    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(dataloader):
            predictions = model(batch_X)
            
            error = torch.abs(predictions[:, 0] - batch_y[:, 0])
            prediction_errors.extend(error.cpu().numpy())
            original_values.extend(batch_y[:, 0].cpu().numpy())

    errors = np.array(prediction_errors)
    threshold = np.mean(errors) + threshold_multiplier * np.std(errors)
    anomalies = errors > threshold

    anomaly_flags_full = np.zeros(len(original_df), dtype=bool)
    start_prediction_indices = np.arange(len(original_df) - SEQUENCE_LENGTH - PREDICTION_LENGTH + 1) + SEQUENCE_LENGTH
    
    valid_anomaly_indices = start_prediction_indices[anomalies]
    valid_anomaly_indices = valid_anomaly_indices[valid_anomaly_indices < len(original_df)]
    
    anomaly_flags_full[valid_anomaly_indices] = True

    return anomaly_flags_full, errors, threshold

# ## TFT Execution

# Now, let\'s execute the TFT model for anomaly detection.

print("\n--- Running TFT Anomaly Detection ---")
try:
    df_tft = pl.read_csv("/home/ubuntu/preprocessed_smart_manufacturing_data.csv")
    if "timestamp" in df_tft.columns and df_tft["timestamp"].dtype != pl.Datetime:
        df_tft = df_tft.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"))

    features_tft = [
        "temperature",
        "vibration",
        "humidity",
        "pressure",
        "energy_consumption"
    ]

    print(f"Preparing data for TFT using features: {features_tft}...")
    X_tensor_tft, y_tensor_tft, scaler_tft = prepare_data_tft(df_tft, features_tft, SEQUENCE_LENGTH, PREDICTION_LENGTH)

    dataset_tft = TensorDataset(X_tensor_tft, y_tensor_tft)
    dataloader_tft = DataLoader(dataset_tft, batch_size=BATCH_SIZE, shuffle=False)

    input_dim_tft = len(features_tft)
    hidden_dim_tft = 64
    output_dim_tft = PREDICTION_LENGTH
    num_heads_tft = 4
    num_layers_tft = 2

    print("Initializing SimplifiedTFT model...")
    model_tft = SimplifiedTFT(input_dim_tft, hidden_dim_tft, output_dim_tft, num_heads_tft, num_layers_tft)

    print("Starting TFT training loop...")
    optimizer_tft = torch.optim.Adam(model_tft.parameters(), lr=LEARNING_RATE)
    criterion_tft = nn.MSELoss()

    for epoch in range(EPOCHS):
        model_tft.train()
        total_loss_tft = 0
        for batch_X_tft, batch_y_tft in dataloader_tft:
            optimizer_tft.zero_grad()
            predictions_tft = model_tft(batch_X_tft)
            loss_tft = criterion_tft(predictions_tft, batch_y_tft)
            loss_tft.backward()
            optimizer_tft.step()
            total_loss_tft += loss_tft.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss_tft/len(dataloader_tft):.4f}")

    print("Detecting anomalies using TFT...")
    anomalies_tft, errors_tft, threshold_tft = detect_anomalies_tft(model_tft, dataloader_tft, scaler_tft, df_tft, features_tft)

    df_tft_results = df_tft.with_columns(
        pl.Series(name="tft_anomaly", values=anomalies_tft)
    )

    print("\nTFT Anomaly Detection Results (first 10 anomalies):")
    print(df_tft_results.filter(pl.col("tft_anomaly") == True).head(10).to_string())

    df_tft_results.write_csv("/home/ubuntu/tft_anomaly_results.csv")
    print("TFT anomaly results saved to tft_anomaly_results.csv")

    # # TFT Analysis and Insights

    # The TFT model identifies anomalies by comparing its predictions with actual values. Large prediction errors indicate potential anomalies.

    # ### Visualizing TFT Anomalies

    # Let\'s visualize the detected anomalies on the \'temperature\' data, as it\'s often a key indicator.

    plt.figure(figsize=(15, 6))
    plt.plot(df_tft["timestamp"], df_tft["temperature"], label="Temperature")
    plt.scatter(
        df_tft.filter(pl.col("tft_anomaly") == True)["timestamp"],
        df_tft.filter(pl.col("tft_anomaly") == True)["temperature"],
        color=\'red\', label=\'Anomaly (TFT)\', s=50, marker=\'o\'
    )
    plt.title("Temperature with TFT Detected Anomalies")
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ### Recommendations for TFT

    # *   **Feature Engineering**: Incorporate more domain-specific features or external factors that might influence machine behavior.
    # *   **Hyperparameter Tuning**: Optimize the model\'s hyperparameters (e.g., hidden_dim, num_heads, learning_rate) for better performance.
    # *   **Quantile Predictions**: A full TFT provides quantile predictions, which can be used to define dynamic anomaly thresholds based on prediction intervals.

except Exception as e:
    print(f"An error occurred during TFT execution: {e}")


# --- Section 4: Comparative Analysis and Overall Recommendations ---

# # Comparative Analysis and Overall Recommendations

# Both PatchTST and TFT offer unique strengths for anomaly detection in time series data. PatchTST excels at capturing local patterns through its patching mechanism, while TFT provides robust forecasting capabilities that can be leveraged for anomaly detection.

# ## Key Observations

# *   **Anomaly Overlap**: Observe if both models detect similar anomalies. Overlapping detections might indicate more significant or persistent issues.
# *   **Model Strengths**: PatchTST might be more sensitive to sudden, localized changes, while TFT might be better at detecting anomalies that manifest as deviations from expected future trends.

# ## Overall Recommendations

# *   **Combine Models**: Consider an ensemble approach where anomalies detected by either model (or both) are flagged. This can improve the robustness of the anomaly detection system.
# *   **Domain Expertise**: Always validate detected anomalies with domain experts. Their insights are crucial for understanding the root causes and significance of anomalies.
# *   **Alerting System**: Implement an alerting system that triggers notifications when anomalies are detected, allowing for timely intervention.
# *   **Root Cause Analysis**: For persistent or critical anomalies, conduct a deeper root cause analysis to address the underlying issues in the manufacturing process.
# *   **Continuous Monitoring and Retraining**: Anomaly detection models should be continuously monitored for performance and retrained periodically with new data to adapt to evolving normal operating conditions.

# This concludes the anomaly detection analysis. The provided code and insights should serve as a strong foundation for building a robust smart manufacturing monitoring system.


