
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Configuration --- #
SEQUENCE_LENGTH = 96  # Length of input sequences for the model
PREDICTION_LENGTH = 24 # Length of output sequences for the model (for forecasting, not directly used for AD)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# --- PatchTST Model Definition (Simplified for Anomaly Detection) --- #
# This is a simplified representation. A full PatchTST implementation is complex.
# For actual anomaly detection, you'd typically use the reconstruction error or
# prediction error from a forecasting task.

class PatchTST(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim

        # Simple patching: Reshape input into patches
        # In a real PatchTST, this would involve more sophisticated embedding
        self.patch_embedding = nn.Linear(patch_len, hidden_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output layer for reconstruction (for anomaly detection via reconstruction error)
        self.output_layer = nn.Linear(hidden_dim, patch_len)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)

        # Apply patching
        # This simple patching assumes input_dim is 1 (univariate time series)
        # For multivariate, you'd patch each variable or flatten and then patch.
        # Here, we'll assume we're processing one feature at a time or handling patches across features.
        # For simplicity, let's assume input_dim is 1 and we're patching the sequence.
        if self.input_dim > 1:
            raise NotImplementedError("Simplified PatchTST only supports input_dim=1 for patching example.")

        # Reshape for patching: (batch_size, sequence_length)
        x_univariate = x.squeeze(-1) # Remove last dim if input_dim is 1

        patches = []
        for i in range(0, x_univariate.shape[1] - self.patch_len + 1, self.stride):
            patches.append(x_univariate[:, i : i + self.patch_len])
        
        # Stack patches: (batch_size, num_patches, patch_len)
        patches = torch.stack(patches, dim=1)

        # Embed patches: (batch_size, num_patches, hidden_dim)
        embedded_patches = self.patch_embedding(patches)

        # Pass through transformer encoder
        # (batch_size, num_patches, hidden_dim)
        encoded_output = self.transformer_encoder(embedded_patches)

        # Reconstruct patches: (batch_size, num_patches, patch_len)
        reconstructed_patches = self.output_layer(encoded_output)

        # This reconstruction is simplified. For anomaly detection, you'd compare
        # reconstructed_patches with original patches to get reconstruction error.
        return reconstructed_patches

# --- Data Preparation --- #
def prepare_data(df, features, sequence_length):
    data = df[features].to_numpy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    X = []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i : i + sequence_length])
    
    X = np.array(X)
    return torch.tensor(X, dtype=torch.float32), scaler

# --- Anomaly Detection Logic --- #
def detect_anomalies_patchtst(model, dataloader, threshold_multiplier=3):
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for batch_X in dataloader:
            # Assuming batch_X is a tuple, take the first element if it's a dataset
            if isinstance(batch_X, (list, tuple)):
                batch_X = batch_X[0]
            
            reconstructed_patches = model(batch_X)
            
            # For simplicity, let's calculate a mean squared error on the first patch
            # In a real scenario, you'd reconstruct the full sequence and compare.
            # Here, we'll compare the first patch of the input with its reconstruction.
            original_first_patch = batch_X[:, 0, :model.patch_len].unsqueeze(1) # (batch_size, 1, patch_len)
            error = torch.mean((original_first_patch - reconstructed_patches[:, 0, :].unsqueeze(1))**2, dim=[-1, -2])
            reconstruction_errors.extend(error.cpu().numpy())

    errors = np.array(reconstruction_errors)
    # Simple thresholding: mean + threshold_multiplier * std
    threshold = np.mean(errors) + threshold_multiplier * np.std(errors)
    anomalies = errors > threshold
    return anomalies, errors, threshold

# --- Main Execution Flow --- #
if __name__ == "__main__":
    print("Loading preprocessed data...")
    try:
        df = pl.read_csv("/home/ubuntu/preprocessed_smart_manufacturing_data.csv")
        # Ensure timestamp is datetime type if not already
        if "timestamp" in df.columns and df["timestamp"].dtype != pl.Datetime:
            df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"))

        # Select numerical features for anomaly detection
        # Exclude 'machine_id', 'machine_status', 'anomaly_flag', 'predicted_remaining_life', 'downtime_risk', 'maintenance_required'
        # and 'failure_type' as they are either identifiers, targets, or categorical.
        features = [
            "temperature",
            "vibration",
            "humidity",
            "pressure",
            "energy_consumption"
        ]

        # Prepare data for PatchTST
        # For this simplified example, we'll treat each feature independently or select one.
        # Let's pick 'temperature' for a univariate example for patching.
        # For multivariate, you'd need a more complex patching strategy or model input.
        print(f"Preparing data for PatchTST using feature: {features[0]}...")
        X_tensor, scaler = prepare_data(df, [features[0]], SEQUENCE_LENGTH)

        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize PatchTST model
        input_dim = 1 # We are using one feature for this example
        hidden_dim = 64
        num_heads = 4
        num_layers = 2
        patch_len = 16 # Example patch length
        stride = 8 # Example stride

        print("Initializing PatchTST model...")
        model = PatchTST(input_dim, hidden_dim, num_heads, num_layers, patch_len, stride)

        # --- Training Loop (Simplified - typically for reconstruction/forecasting) --- #
        # For anomaly detection, PatchTST is often trained on 'normal' data
        # to learn its patterns. Anomalies are then deviations from these patterns.
        print("Starting simplified training loop...")
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch_X in dataloader:
                if isinstance(batch_X, (list, tuple)):
                    batch_X = batch_X[0]

                optimizer.zero_grad()
                reconstructed_patches = model(batch_X)
                
                # Calculate loss based on reconstruction of the first patch for simplicity
                original_first_patch = batch_X[:, 0, :patch_len].unsqueeze(1)
                loss = criterion(reconstructed_patches[:, 0, :].unsqueeze(1), original_first_patch)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

        # --- Anomaly Detection --- #
        print("Detecting anomalies using PatchTST...")
        anomalies, errors, threshold = detect_anomalies_patchtst(model, dataloader)

        # Map anomalies back to original DataFrame indices
        # Note: The anomaly flags will be for sequences, not individual data points.
        # A more sophisticated mapping is needed for point-wise anomalies.
        anomaly_flags_full = np.zeros(len(df), dtype=bool)
        # This is a simplification. Anomalies are detected on sequences.
        # We'll mark the end of the sequence as anomalous for simplicity.
        anomaly_indices = np.where(anomalies)[0] + SEQUENCE_LENGTH -1
        anomaly_indices = anomaly_indices[anomaly_indices < len(df)]
        anomaly_flags_full[anomaly_indices] = True

        df_results = df.with_columns(
            pl.Series(name="patchtst_anomaly", values=anomaly_flags_full)
        )

        print("\nPatchTST Anomaly Detection Results (first 10 anomalies):")
        print(df_results.filter(pl.col("patchtst_anomaly") == True).head(10).to_string())

        # Save results
        df_results.write_csv("/home/ubuntu/patchtst_anomaly_results.csv")
        print("PatchTST anomaly results saved to patchtst_anomaly_results.csv")

    except FileNotFoundError:
        print("Error: preprocessed_smart_manufacturing_data.csv not found. Please run data_preprocessing_polars.py first.")
    except Exception as e:
        print(f"An error occurred during PatchTST execution: {e}")



