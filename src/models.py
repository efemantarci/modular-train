import torch
import torch.nn as nn

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