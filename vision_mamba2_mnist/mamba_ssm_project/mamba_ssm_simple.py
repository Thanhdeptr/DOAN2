import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleMambaSSM(nn.Module):
    """
    Simplified Mamba SSM implementation
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.B = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        
        # Convolution
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, d_model = x.shape
        
        # Project input
        x_proj = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_proj = x_proj.chunk(2, dim=-1)
        x_proj, gate = x_proj[0], x_proj[1]  # (batch, seq_len, d_inner)
        
        # Apply convolution
        x_conv = self.conv(x_proj.transpose(-1, -2))  # (batch, d_inner, seq_len)
        x_conv = x_conv.transpose(-1, -2)  # (batch, seq_len, d_inner)
        x_conv = x_conv[:, :seq_len, :]  # Remove padding
        
        # Apply gate
        x_conv = x_conv * F.silu(gate)
        
        # Simplified SSM computation
        # Initialize state
        state = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # Update state: s_t = A * s_{t-1} + B * x_t
            state = torch.einsum('bds,ds->bds', state, self.A) + \
                   torch.einsum('bd,ds->bds', x_conv[:, t, :], self.B)
            
            # Output: y_t = C * s_t + D * x_t
            output = torch.einsum('bds,ds->bd', state, self.C) + \
                    self.D * x_conv[:, t, :]
            
            outputs.append(output.unsqueeze(1))
        
        # Concatenate outputs
        y = torch.cat(outputs, dim=1)  # (batch, seq_len, d_inner)
        
        # Project output
        y = self.out_proj(y)  # (batch, seq_len, d_model)
        
        # Residual connection and normalization
        y = self.norm(y + x)
        
        return y


class MambaBlock(nn.Module):
    """
    Mamba Block với SSM
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        # Mamba SSM
        self.mamba = SimpleMambaSSM(d_model, d_state, d_conv, expand)
        
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
    Vision Mamba 2 Architecture với Mamba SSM thực sự
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
            MambaBlock(d_model=d_model) for _ in range(num_blocks)
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
    
    # Test SSM
    print("\nTesting SimpleMambaSSM...")
    ssm = SimpleMambaSSM(d_model=128)
    x_ssm = torch.randn(4, 14, 128)  # (batch, seq_len, d_model)
    y_ssm = ssm(x_ssm)
    print(f"SSM Input: {x_ssm.shape}")
    print(f"SSM Output: {y_ssm.shape}")
    
    print("\n✅ Mamba SSM implementation successful!")
