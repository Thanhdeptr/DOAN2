import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionSimple(nn.Module):
    """
    Vision Simple Architecture cho MNIST
    Sử dụng CNN + LSTM đơn giản
    """
    def __init__(self, num_classes=10, d_model=128):
        super().__init__()
        
        self.d_model = d_model
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, d_model, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(d_model)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # Input: (batch, 1, 28, 28) cho MNIST
        
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 32, 28, 28)
        x = self.pool(x)  # (batch, 32, 14, 14)
        
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 64, 14, 14)
        x = self.pool(x)  # (batch, 64, 7, 7)
        
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, d_model, 7, 7)
        
        # Reshape cho LSTM
        batch, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        x = x.reshape(batch, height * width, channels)  # (batch, 49, d_model)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Lấy output cuối cùng
        x = lstm_out[:, -1, :]  # (batch, d_model)
        
        # Classification
        x = self.classifier(x)
        
        return x


def create_vision_simple_model(num_classes=10, d_model=128):
    """
    Factory function để tạo Vision Simple model
    """
    return VisionSimple(
        num_classes=num_classes,
        d_model=d_model
    )


if __name__ == "__main__":
    # Test model
    model = create_vision_simple_model()
    x = torch.randn(2, 1, 28, 28)  # 2 samples, 1 channel, 28x28 (MNIST size)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

