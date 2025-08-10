import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleMambaBlock(nn.Module):
    """
    Simple Mamba Block - sử dụng LSTM thay thế cho Mamba SSM
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # LSTM layer thay thế cho Mamba
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch, channels, height, width = x.shape
        
        # Reshape để LSTM có thể xử lý
        # Chuyển thành (batch * height, width, channels)
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        x = x.reshape(batch * height, width, channels)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Lấy output cuối cùng và reshape
        x = lstm_out[:, -1, :]  # (batch * height, channels)
        x = x.reshape(batch, height, channels)
        x = x.unsqueeze(3)  # (batch, height, channels, 1)
        x = x.permute(0, 2, 1, 3)  # (batch, channels, height, 1)
        
        # Expand để match với input width
        x = x.expand(-1, -1, -1, width)
        
        # Apply normalization
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x


class VisionMamba2Simple(nn.Module):
    """
    Vision Mamba 2 Simple Architecture cho MNIST
    Sử dụng LSTM thay thế cho Mamba SSM
    """
    def __init__(self, num_classes=10, d_model=128, num_blocks=4):
        super().__init__()
        
        self.d_model = d_model
        self.num_blocks = num_blocks
        
        # Initial convolution để giảm kích thước
        self.input_conv = nn.Conv2d(1, d_model, kernel_size=7, stride=2, padding=3)
        self.input_norm = nn.BatchNorm2d(d_model)
        
        # Simple Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            SimpleMambaBlock(d_model=d_model) for _ in range(num_blocks)
        ])
        
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
        
        # Initial convolution
        x = self.input_conv(x)  # (batch, d_model, 14, 14)
        x = self.input_norm(x)
        x = F.relu(x)
        
        # Apply Mamba blocks
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch, d_model, 1, 1)
        x = x.flatten(1)  # (batch, d_model)
        
        # Classification
        x = self.classifier(x)
        
        return x


def create_vision_mamba2_simple_model(num_classes=10, d_model=128, num_blocks=4):
    """
    Factory function để tạo Vision Mamba 2 Simple model
    """
    return VisionMamba2Simple(
        num_classes=num_classes,
        d_model=d_model,
        num_blocks=num_blocks
    )


if __name__ == "__main__":
    # Test model
    model = create_vision_mamba2_simple_model()
    x = torch.randn(2, 1, 28, 28)  # 2 samples, 1 channel, 28x28 (MNIST size)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
