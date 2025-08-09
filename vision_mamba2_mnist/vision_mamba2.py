import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math


class VisionMambaBlock(nn.Module):
    """
    Vision Mamba Block - thay thế cho Conv2D trong CNN
    Sử dụng Mamba SSM để xử lý spatial information
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        d_inner = int(expand * d_model)
        
        # Mamba SSM layer
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch, channels, height, width = x.shape
        
        # Reshape để Mamba có thể xử lý
        # Chuyển thành (batch * height, width, channels)
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        x = x.reshape(batch * height, width, channels)
        
        # Apply Mamba
        x = self.mamba(x)
        
        # Reshape lại về dạng ban đầu
        x = x.reshape(batch, height, width, channels)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width)
        
        # Apply normalization
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x


class VisionMamba2(nn.Module):
    """
    Vision Mamba 2 Architecture cho MNIST
    Thay thế CNN bằng Mamba SSM blocks
    """
    def __init__(self, num_classes=10, d_model=128, num_blocks=4):
        super().__init__()
        
        self.d_model = d_model
        self.num_blocks = num_blocks
        
        # Initial convolution để giảm kích thước
        self.input_conv = nn.Conv2d(1, d_model, kernel_size=7, stride=2, padding=3)
        self.input_norm = nn.BatchNorm2d(d_model)
        
        # Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            VisionMambaBlock(d_model=d_model) for _ in range(num_blocks)
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


def create_vision_mamba2_model(num_classes=10, d_model=128, num_blocks=4):
    """
    Factory function để tạo Vision Mamba 2 model
    """
    return VisionMamba2(
        num_classes=num_classes,
        d_model=d_model,
        num_blocks=num_blocks
    )


if __name__ == "__main__":
    # Test model
    model = create_vision_mamba2_model()
    x = torch.randn(2, 1, 28, 28)  # 2 samples, 1 channel, 28x28 (MNIST size)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}") 