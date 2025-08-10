# Vision Mamba 2 với Mamba SSM

Dự án này implement **Mamba SSM (State Space Model)** cho vision tasks, cụ thể là classification trên MNIST dataset. Đây là một implementation từ đầu của Mamba SSM dựa trên nghiên cứu từ paper và video.

## 🚀 Mamba SSM là gì?

**Mamba SSM** là một kiến trúc mới thay thế cho Transformer, sử dụng **State Space Models** để xử lý sequences. Ưu điểm chính:

- **Linear complexity**: O(n) thay vì O(n²) như Transformer
- **Long sequence handling**: Có thể xử lý sequences dài hơn hiệu quả
- **Selective scanning**: Chỉ focus vào những phần quan trọng của input
- **Memory efficient**: Ít memory hơn so với attention mechanisms

## 📁 Cấu trúc Project

```
vision_mamba2_mnist/
├── mamba_ssm_simple.py      # Implementation Mamba SSM từ đầu
├── train_mamba.py           # Script training cho Mamba SSM
├── demo_mamba.py            # Demo và analysis
├── mamba_ssm.py             # Phiên bản Mamba SSM đầu tiên (có lỗi)
├── vision_simple.py         # Model LSTM cũ (backup)
├── train_simple.py          # Training script cho LSTM
├── demo.py                  # Demo cho LSTM
├── requirements.txt         # Dependencies
├── README.md               # README cũ
└── README_MAMBA.md         # README này
```

## 🔧 Installation

```bash
# Kích hoạt virtual environment
myenv\Scripts\activate

# Cài đặt dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm tensorboard scikit-learn seaborn
```

## 🏗️ Kiến trúc Mamba SSM

### 1. SelectiveScan Module
```python
class SelectiveScan(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        # SSM parameters: A, B, C, D
        # Convolution layer cho local interactions
        # Projection layers
```

**Các thành phần chính:**
- **SSM Parameters**: A, B, C, D matrices cho state space modeling
- **Convolution**: Xử lý local interactions
- **Gating**: Selective information flow
- **State Update**: s_t = A * s_{t-1} + B * x_t
- **Output**: y_t = C * s_t + D * x_t

### 2. MambaBlock
```python
class MambaBlock(nn.Module):
    def forward(self, x):
        # Reshape 2D -> 1D sequence
        # Apply SelectiveScan
        # Reshape back to 2D
```

### 3. VisionMamba2
```python
class VisionMamba2(nn.Module):
    def __init__(self, num_classes=10, d_model=128, num_blocks=4):
        # Initial convolution
        # Multiple Mamba blocks
        # Classification head
```

## 🚀 Usage

### 1. Test Implementation
```bash
python mamba_ssm_simple.py
```

### 2. Training
```bash
python train_mamba.py
```

**Hyperparameters:**
- Batch size: 64
- Learning rate: 0.001
- Epochs: 20
- Model size: 128 dimensions
- Number of blocks: 4

### 3. Demo và Analysis
```bash
python demo_mamba.py
```

**Các tính năng:**
- Performance analysis
- Confusion matrix
- Prediction visualization
- Interactive demo

## 📊 Kết quả mong đợi

Với Mamba SSM implementation này, bạn có thể mong đợi:

- **Accuracy**: ~95-98% trên MNIST test set
- **Training time**: Nhanh hơn Transformer tương đương
- **Memory usage**: Thấp hơn attention-based models
- **Scalability**: Có thể scale lên sequences dài hơn

## 🔬 So sánh với các kiến trúc khác

| Architecture | Complexity | Memory | Long Sequences | Implementation |
|--------------|------------|--------|----------------|----------------|
| Transformer  | O(n²)      | High   | Limited        | Attention      |
| LSTM         | O(n)       | Medium | Good           | Recurrent      |
| **Mamba SSM**| **O(n)**   | **Low**| **Excellent**  | **State Space**|

## 🛠️ Technical Details

### State Space Model
Mamba SSM sử dụng linear state space model:

```
Ẩn state: s_t = A * s_{t-1} + B * x_t
Output:   y_t = C * s_t + D * x_t
```

### Selective Scanning
- Chỉ process những phần quan trọng của input
- Sử dụng gating mechanism để control information flow
- Convolution layer cho local context

### Reshaping Strategy
```python
# 2D -> 1D sequence
x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
x = x.reshape(batch * height, width, channels)

# Apply Mamba SSM
x = self.mamba(x)

# 1D -> 2D back
x = x.reshape(batch, height, width, channels)
x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width)
```

## 🎯 Nghiên cứu và References

Dự án này dựa trên:
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)
- Video tutorial: [Mamba SSM Explained](https://www.youtube.com/watch?v=HnRBLD3_k7g)

## 🔄 Migration từ LSTM

Nếu bạn muốn chuyển từ LSTM sang Mamba SSM:

1. **Backup LSTM model**: Đã có trong branch `LSTM-backup`
2. **Test Mamba SSM**: `python mamba_ssm_simple.py`
3. **Train Mamba SSM**: `python train_mamba.py`
4. **Compare results**: Sử dụng demo scripts

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory**: Giảm batch size
2. **Slow training**: Kiểm tra device (CPU/GPU)
3. **Import errors**: Kiểm tra virtual environment

### Performance tips:

1. **Batch size**: 32-64 cho GPU, 16-32 cho CPU
2. **Learning rate**: 0.001 với AdamW optimizer
3. **Model size**: 128 dimensions cho MNIST

## 📈 Next Steps

1. **Scale up**: Thử nghiệm với datasets lớn hơn (CIFAR-10, ImageNet)
2. **Architecture improvements**: Thêm residual connections, layer normalization
3. **Hyperparameter tuning**: Grid search cho optimal parameters
4. **Multi-modal**: Kết hợp với text hoặc audio

## 🤝 Contributing

Để contribute vào project:

1. Fork repository
2. Tạo feature branch
3. Implement changes
4. Test thoroughly
5. Submit pull request

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

---

**Happy coding với Mamba SSM! 🚀**
