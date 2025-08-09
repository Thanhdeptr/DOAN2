#!/usr/bin/env python3
"""
Quick test script để kiểm tra Vision Mamba 2 setup
"""

import torch
import torch.nn as nn
from vision_mamba2 import create_vision_mamba2_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def test_imports():
    """Test các imports cần thiết"""
    print("Testing imports...")
    
    try:
        from mamba_ssm import Mamba
        print("✓ Mamba SSM imported successfully")
    except ImportError as e:
        print(f"✗ Mamba SSM import failed: {e}")
        return False
    
    try:
        import torch
        import torchvision
        print("✓ PyTorch and torchvision imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test tạo model"""
    print("\nTesting model creation...")
    
    try:
        model = create_vision_mamba2_model()
        print("✓ Model created successfully")
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {total_params:,} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_data_loading():
    """Test loading MNIST data"""
    print("\nTesting data loading...")
    
    try:
        # Create a small test dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load just a few samples
        dataset = datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Test one batch
        for batch_idx, (data, target) in enumerate(dataloader):
            print(f"✓ Data loaded successfully: {data.shape}, targets: {target.shape}")
            break
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_training_step():
    """Test một training step"""
    print("\nTesting training step...")
    
    try:
        # Create model
        model = create_vision_mamba2_model()
        
        # Create optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create dummy data
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful: Loss = {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return False


def main():
    """Main test function"""
    print("=== Vision Mamba 2 Quick Test ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Model Creation Test", test_model_creation),
        ("Data Loading Test", test_data_loading),
        ("Training Step Test", test_training_step)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print("=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to train Vision Mamba 2.")
        print("\nNext steps:")
        print("1. Run: python train.py")
        print("2. After training, run: python test_model.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    main() 