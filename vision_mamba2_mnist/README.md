# Vision Mamba 2 - MNIST Classification

Dự án này implement Vision Mamba 2 architecture để phân loại MNIST digits, thay thế CNN bằng Mamba SSM blocks.

## Cấu trúc dự án

```
vision_mamba2_mnist/
├── vision_mamba2.py      # Vision Mamba 2 architecture
├── train.py              # Training script
├── test_model.py         # Testing và visualization
├── requirements.txt      # Dependencies
└── README.md            # Hướng dẫn này
```

## Cài đặt

1. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Kiểm tra mamba-ssm:**
```bash
python -c "from mamba_ssm import Mamba; print('Mamba SSM installed successfully!')"
```

## Sử dụng

### 1. Training Model

```bash
python train.py
```

**Hyperparameters có thể điều chỉnh trong `train.py`:**
- `batch_size`: 32
- `num_epochs`: 10
- `learning_rate`: 0.001
- `d_model`: 128 (kích thước hidden dimension)
- `num_blocks`: 4 (số lượng Mamba blocks)

### 2. Testing và Visualization

```bash
python test_model.py
```

Script này sẽ:
- Load model đã train
- Phân tích performance
- Vẽ confusion matrix
- Visualize predictions

### 3. Test Architecture

```bash
python vision_mamba2.py
```

Kiểm tra model architecture và parameters.

## Architecture

### VisionMambaBlock
- **Mục đích:** Thay thế Conv2D layers bằng Mamba SSM
- **Input:** (batch, channels, height, width)
- **Process:** Reshape → Mamba SSM → Reshape back
- **Output:** (batch, channels, height, width)

### VisionMamba2
- **Input Conv:** Giảm kích thước từ 28x28 → 14x14
- **Mamba Blocks:** 4 blocks xử lý spatial information
- **Global Pooling:** Adaptive average pooling
- **Classifier:** 2-layer MLP với dropout

## Kết quả mong đợi

- **Accuracy:** ~95-98% trên MNIST test set
- **Training time:** ~5-10 phút trên GPU
- **Model size:** ~500K parameters

## Files được tạo

- `vision_mamba2_mnist.pth`: Trained model weights
- `training_history.png`: Biểu đồ training progress
- `confusion_matrix.png`: Confusion matrix
- `predictions_visualization.png`: Sample predictions
- `runs/`: TensorBoard logs

## So sánh với CNN

| Aspect | CNN | Vision Mamba 2 |
|--------|-----|----------------|
| Architecture | Convolutional layers | Mamba SSM blocks |
| Memory efficiency | O(n²) | O(n) |
| Parallelization | Limited | Better |
| Long-range dependencies | Limited | Better |

## Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory:**
   - Giảm `batch_size`
   - Giảm `d_model`

2. **Mamba SSM not found:**
   ```bash
   pip install mamba-ssm
   ```

3. **Slow training:**
   - Sử dụng GPU
   - Giảm `num_workers` trong DataLoader

## Tùy chỉnh

### Thay đổi model size:
```python
model = create_vision_mamba2_model(
    d_model=256,  # Tăng từ 128
    num_blocks=6  # Tăng từ 4
)
```

### Thay đổi dataset:
Chỉnh sửa `get_mnist_dataloaders()` trong `train.py` để sử dụng dataset khác.

## References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417) 