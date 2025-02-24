import torch
import torch.nn as nn
import math
class Temporal3DPredictorAndDistance(nn.Module):
    # Predicts 2 3D points and their distances to origin (8 values). Kept for compatibility with old models.
    def __init__(self, spatial_bins=20, hidden_dim=512):
        super(Temporal3DPredictorAndDistance, self).__init__()
        # Convolutional layers to capture spatial features
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # LSTM to handle temporal sequence information
        self.lstm = nn.LSTM(input_size=128 * ((spatial_bins // 8) ** 3), hidden_size=hidden_dim, num_layers=2, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 8)  # Predict 2 3D points and their distances to origin (8 values)

    def process_time_step(self, x_t):
        x_t = nn.functional.relu(self.conv1(x_t))  # (batch_size, 128, x, y, z)
        x_t = self.pool1(x_t)                      # (batch_size, 128, x/2, y/2, z/2)
        x_t = nn.functional.relu(self.conv2(x_t))
        x_t = self.pool2(x_t)
        x_t = nn.functional.relu(self.conv3(x_t))
        x_t = self.pool3(x_t)
        x_t = nn.functional.relu(self.conv4(x_t))
        x_t = x_t.view(x_t.size(0), -1)            # Flatten (batch_size, 1024)
        return x_t

    def forward(self, x):
        batch_size, time_steps, _, _, _ = x.shape
        
        # Process each time step sequentially
        conv_features = []
        for t in range(time_steps):
            x_t = x[:, t, :, :, :].unsqueeze(1)  # (batch_size, 1, x, y, z)
            conv_features.append(self.process_time_step(x_t))
        
        conv_features = torch.stack(conv_features, dim=1)
        
        # Pass through LSTM for temporal processing
        lstm_out, _ = self.lstm(conv_features)  # Shape: (batch_size, time_steps, hidden_dim)
        
        # Take only the last time step output
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
        
        # Fully connected layers
        x = nn.functional.relu(self.fc1(lstm_out))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
class Temporal3DPredictor(nn.Module):
    # Predicts output_size values.
    def __init__(self, spatial_bins=20, output_size=8, hidden_dim=512):
        super(Temporal3DPredictor, self).__init__()
        # Convolutional layers to capture spatial features
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # LSTM to handle temporal sequence information
        self.lstm = nn.LSTM(input_size=128 * ((spatial_bins // 8) ** 3), hidden_size=hidden_dim, num_layers=2, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

    def process_time_step(self, x_t):
        x_t = nn.functional.relu(self.conv1(x_t))  # (batch_size, 128, x, y, z)
        x_t = self.pool1(x_t)                      # (batch_size, 128, x/2, y/2, z/2)
        x_t = nn.functional.relu(self.conv2(x_t))
        x_t = self.pool2(x_t)
        x_t = nn.functional.relu(self.conv3(x_t))
        x_t = self.pool3(x_t)
        x_t = nn.functional.relu(self.conv4(x_t))
        x_t = x_t.view(x_t.size(0), -1)            # Flatten (batch_size, 1024)
        return x_t

    def forward(self, x):
        batch_size, time_steps, _, _, _ = x.shape
        
        # Process each time step sequentially
        conv_features = []
        for t in range(time_steps):
            x_t = x[:, t, :, :, :].unsqueeze(1)  # (batch_size, 1, x, y, z)
            conv_features.append(self.process_time_step(x_t))
        
        conv_features = torch.stack(conv_features, dim=1)
        
        # Pass through LSTM for temporal processing
        lstm_out, _ = self.lstm(conv_features)  # Shape: (batch_size, time_steps, hidden_dim)
        
        # Take only the last time step output
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
        
        # Fully connected layers
        x = nn.functional.relu(self.fc1(lstm_out))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
class Temporal3DPredictorAndDistanceDropout(nn.Module):
    # Predicts 2 3D points and their distances to origin (8 values). Kept for compatibility with old models.
    def __init__(self, spatial_bins=20, hidden_dim=512, dropout=0.2):
        super(Temporal3DPredictorAndDistanceDropout, self).__init__()
        # Convolutional layers to capture spatial features
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # LSTM to handle temporal sequence information
        self.lstm = nn.LSTM(input_size=128 * ((spatial_bins // 8) ** 3), hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(128, 8)  # Predict 2 3D points and their distances to origin (8 values)

    def process_time_step(self, x_t):
        x_t = nn.functional.relu(self.conv1(x_t))  # (batch_size, 128, x, y, z)
        x_t = self.pool1(x_t)                      # (batch_size, 128, x/2, y/2, z/2)
        x_t = nn.functional.relu(self.conv2(x_t))
        x_t = self.pool2(x_t)
        x_t = nn.functional.relu(self.conv3(x_t))
        x_t = self.pool3(x_t)
        x_t = nn.functional.relu(self.conv4(x_t))
        x_t = x_t.view(x_t.size(0), -1)            # Flatten (batch_size, 1024)
        return x_t

    def forward(self, x):
        batch_size, time_steps, _, _, _ = x.shape
        
        # Process each time step sequentially
        conv_features = []
        for t in range(time_steps):
            x_t = x[:, t, :, :, :].unsqueeze(1)  # (batch_size, 1, x, y, z)
            conv_features.append(self.process_time_step(x_t))
        
        conv_features = torch.stack(conv_features, dim=1)
        
        # Pass through LSTM for temporal processing
        lstm_out, _ = self.lstm(conv_features)  # Shape: (batch_size, time_steps, hidden_dim)
        
        # Take only the last time step output
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
        
        # Fully connected layers
        x = nn.functional.relu(self.fc1(lstm_out))
        x = self.dropout1(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class Temporal3DPredictorDropout(nn.Module):
    # Predicts output_size values.
    def __init__(self, spatial_bins=20, output_size=8, hidden_dim=512, dropout=0.2):
        super(Temporal3DPredictorDropout, self).__init__()
        # Convolutional layers to capture spatial features
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # LSTM to handle temporal sequence information
        self.lstm = nn.LSTM(input_size=128 * ((spatial_bins // 8) ** 3), hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(128, output_size)

    def process_time_step(self, x_t):
        x_t = nn.functional.relu(self.conv1(x_t))  # (batch_size, 128, x, y, z)
        x_t = self.pool1(x_t)                      # (batch_size, 128, x/2, y/2, z/2)
        x_t = nn.functional.relu(self.conv2(x_t))
        x_t = self.pool2(x_t)
        x_t = nn.functional.relu(self.conv3(x_t))
        x_t = self.pool3(x_t)
        x_t = nn.functional.relu(self.conv4(x_t))
        x_t = x_t.view(x_t.size(0), -1)            # Flatten (batch_size, 1024)
        return x_t

    def forward(self, x):
        batch_size, time_steps, _, _, _ = x.shape
        
        # Process each time step sequentially
        conv_features = []
        for t in range(time_steps):
            x_t = x[:, t, :, :, :].unsqueeze(1)  # (batch_size, 1, x, y, z)
            conv_features.append(self.process_time_step(x_t))
        
        conv_features = torch.stack(conv_features, dim=1)
        
        # Pass through LSTM for temporal processing
        lstm_out, _ = self.lstm(conv_features)  # Shape: (batch_size, time_steps, hidden_dim)
        
        # Take only the last time step output
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
        
        # Fully connected layers
        x = nn.functional.relu(self.fc1(lstm_out))
        x = self.dropout1(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.register_buffer('pe', self._build_pe(d_model, max_len))
        
    def _build_pe(self, d_model, max_len):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Temporal3DTransformer(nn.Module):
    def __init__(self, spatial_bins=20, output_size=8, d_model=512, dropout=0.2):
        super().__init__()
        # Original CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Transformer components
        self.conv_proj = nn.Linear(128*(spatial_bins//8)**3, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=2048,
                dropout=dropout
            ),
            num_layers=2
        )
        
        # Same FC layers as original
        self.fc = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        batch_size, time_steps = x.shape[:2]
        cnn_features = []
        
        for t in range(time_steps):
            x_t = x[:, t].unsqueeze(1)  # [2, 1, 20, 20, 20]
            features = self.conv_layers(x_t).flatten(1)  # [2, 1024]
            cnn_features.append(features)
            
        x = torch.stack(cnn_features, dim=1)  # [2, 5, 1024]
        x = self.conv_proj(x)  # [2, 5, 512]
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, features]
        x = self.transformer(x)[-1]  # Take last timestep [2, 512]
        return self.fc(x)