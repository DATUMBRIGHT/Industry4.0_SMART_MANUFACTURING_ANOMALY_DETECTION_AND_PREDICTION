
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Configuration --- #
SEQUENCE_LENGTH = 96  # Length of input sequences for the model
PREDICTION_LENGTH = 24 # Length of output sequences for the model (for forecasting)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# --- Simplified TFT Model Definition --- #
# This is a highly simplified version of a Temporal Fusion Transformer (TFT).
# A full TFT implementation involves many components like: 
# - Gated Residual Networks (GRN)
# - Variable Selection Networks
# - Gated skip connections
# - Multi-head attention
# - Quantile regression output
# For anomaly detection, we'll focus on its forecasting capability and use prediction error.

class SimplifiedTFT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input processing (simple linear layer for embedding)
        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        # Encoder (similar to a simple Transformer Encoder)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Decoder (simple linear layer for forecasting)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x): # x shape: (batch_size, sequence_length, input_dim)
        # Embed input
        embedded_input = self.input_embedding(x)

        # Pass through encoder
        encoded_output = self.transformer_encoder(embedded_input)

        # Take the last hidden state for prediction (simplified)
        # In a real TFT, this would involve more complex attention mechanisms
        # and potentially static covariates.
        last_hidden_state = encoded_output[:, -1, :]

        # Predict future values
        predictions = self.decoder(last_hidden_state)

        return predictions

# --- Data Preparation --- #
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

# --- Anomaly Detection Logic --- #
def detect_anomalies_tft(model, dataloader, scaler, original_df, features, threshold_multiplier=3):
    model.eval()
    prediction_errors = []
    original_values = []
    
    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(dataloader):
            predictions = model(batch_X)
            
            # Calculate error for the first predicted step of the first feature
            error = torch.abs(predictions[:, 0] - batch_y[:, 0])
            prediction_errors.extend(error.cpu().numpy())
            original_values.extend(batch_y[:, 0].cpu().numpy())

    errors = np.array(prediction_errors)
    # Simple thresholding: mean + threshold_multiplier * std
    threshold = np.mean(errors) + threshold_multiplier * np.std(errors)
    anomalies = errors > threshold

    # Map anomalies back to original DataFrame indices
    # The anomaly flag will correspond to the first predicted timestamp
    anomaly_flags_full = np.zeros(len(original_df), dtype=bool)
    # The index in the original dataframe corresponding to the start of the prediction window
    start_prediction_indices = np.arange(len(original_df) - SEQUENCE_LENGTH - PREDICTION_LENGTH + 1) + SEQUENCE_LENGTH
    
    # Filter for indices that are within the bounds of the original dataframe
    valid_anomaly_indices = start_prediction_indices[anomalies]
    valid_anomaly_indices = valid_anomaly_indices[valid_anomaly_indices < len(original_df)]
    
    anomaly_flags_full[valid_anomaly_indices] = True

    return anomaly_flags_full, errors, threshold

# --- Main Execution Flow --- #
if __name__ == "__main__":
    print("Loading preprocessed data...")
    try:
        df = pl.read_csv("/home/ubuntu/preprocessed_smart_manufacturing_data.csv")
        # Ensure timestamp is datetime type if not already
        if "timestamp" in df.columns and df["timestamp"].dtype != pl.Datetime:
            df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"))

        # Select numerical features for anomaly detection
        features = [
            "temperature",
            "vibration",
            "humidity",
            "pressure",
            "energy_consumption"
        ]

        # Prepare data for TFT
        print(f"Preparing data for TFT using features: {features}...")
        X_tensor, y_tensor, scaler = prepare_data_tft(df, features, SEQUENCE_LENGTH, PREDICTION_LENGTH)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize SimplifiedTFT model
        input_dim = len(features)
        hidden_dim = 64
        output_dim = PREDICTION_LENGTH # Predicting future values
        num_heads = 4
        num_layers = 2

        print("Initializing SimplifiedTFT model...")
        model = SimplifiedTFT(input_dim, hidden_dim, output_dim, num_heads, num_layers)

        # --- Training Loop --- #
        print("Starting TFT training loop...")
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

        # --- Anomaly Detection --- #
        print("Detecting anomalies using TFT...")
        anomalies, errors, threshold = detect_anomalies_tft(model, dataloader, scaler, df, features)

        df_results = df.with_columns(
            pl.Series(name="tft_anomaly", values=anomalies)
        )

        print("\nTFT Anomaly Detection Results (first 10 anomalies):")
        print(df_results.filter(pl.col("tft_anomaly") == True).head(10).to_string())

        # Save results
        df_results.write_csv("/home/ubuntu/tft_anomaly_results.csv")
        print("TFT anomaly results saved to tft_anomaly_results.csv")

    except FileNotFoundError:
        print("Error: preprocessed_smart_manufacturing_data.csv not found. Please run data_preprocessing_polars.py first.")
    except Exception as e:
        print(f"An error occurred during TFT execution: {e}")



