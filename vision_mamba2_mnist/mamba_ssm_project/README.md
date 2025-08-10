# Mamba SSM Project

Dự án implement **Mamba SSM (State Space Model)** cho vision tasks trên MNIST dataset.

## 📁 Cấu trúc Project

```
mamba_ssm_project/
├── mamba_ssm_simple.py      # ✅ Implementation Mamba SSM hoạt động tốt
├── train_mamba.py           # ✅ Script training cho Mamba SSM
├── demo_mamba.py            # ✅ Demo và analysis
├── best_mamba_mnist.pth     # ✅ Model đã train (97.85% accuracy)
└── README_MAMBA.md          # 📖 Documentation chi tiết
```

## 🚀 Quick Start

### 1. Test Implementation
```bash
cd mamba_ssm_project
python mamba_ssm_simple.py
```

### 2. Training (nếu cần)
```bash
python train_mamba.py
```

### 3. Demo và Analysis
```bash
python demo_mamba.py
```

## 📊 Kết quả

- **Test Accuracy**: 97.85%
- **Model Parameters**: 466,122
- **Training Time**: ~8 epochs (đã đủ tốt)

## 🔧 Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm tensorboard scikit-learn seaborn
```

## 🎯 Features

- **Mamba SSM Implementation**: State Space Model từ đầu
- **Vision Architecture**: CNN + Mamba blocks
- **MNIST Classification**: 10 classes
- **Interactive Demo**: Test với bất kỳ ảnh nào
- **Performance Analysis**: Confusion matrix, classification report

## 📈 Performance

| Metric | Value |
|--------|-------|
| Accuracy | 97.85% |
| Precision | 98% |
| Recall | 98% |
| F1-Score | 98% |

## 🗂️ Files Explanation

- **`mamba_ssm_simple.py`**: ✅ File chính - implementation Mamba SSM hoạt động tốt
- **`train_mamba.py`**: ✅ Training script với TensorBoard logging
- **`demo_mamba.py`**: ✅ Demo với 4 options: performance, confusion matrix, visualization, interactive
- **`best_mamba_mnist.pth`**: ✅ Model weights đã train
- **`README_MAMBA.md`**: 📖 Documentation chi tiết về kiến trúc và technical details

## 🚫 Files Removed

- **`mamba_ssm.py`**: ❌ Đã xóa - có lỗi einsum và d_inner variable
- **`mamba_ssm_mnist_interrupted.pth`**: ❌ Model interrupted - không cần thiết

## 🎉 Success!

Mamba SSM implementation đã hoạt động thành công với accuracy 97.85% trên MNIST test set!
