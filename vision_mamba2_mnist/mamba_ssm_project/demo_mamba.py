import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from mamba_ssm_simple import create_vision_mamba2_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def load_trained_model(model_path, device):
    """
    Load trained Mamba SSM model
    """
    model = create_vision_mamba2_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def analyze_performance(model, test_loader, device):
    """
    Analyze model performance
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100. * sum(1 for x, y in zip(all_predictions, all_targets) if x == y) / len(all_targets)
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions))
    
    return all_predictions, all_targets, all_probabilities, accuracy


def plot_confusion_matrix(all_targets, all_predictions):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(all_targets, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - Mamba SSM on MNIST')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('mamba_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(model, test_loader, device, num_samples=16):
    """
    Visualize model predictions
    """
    model.eval()
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            # Get the first sample from batch
            img = data[0].cpu().squeeze()
            true_label = target[0].cpu().item()
            pred_label = predicted[0].cpu().item()
            confidence = probabilities[0][pred_label].cpu().item()
            
            # Plot
            axes[i].imshow(img, cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}', 
                            color=color, fontsize=10)
            axes[i].axis('off')
    
    plt.suptitle('Mamba SSM Predictions on MNIST', fontsize=16)
    plt.tight_layout()
    plt.savefig('mamba_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def interactive_demo(model, device):
    """
    Interactive demo để test model với input từ user
    """
    print("\n=== Interactive Demo ===")
    print("Nhập 'quit' để thoát")
    
    # Load test dataset để lấy samples
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    while True:
        try:
            user_input = input("\nNhập index của ảnh (0-9999) hoặc 'quit': ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            idx = int(user_input)
            if idx < 0 or idx >= len(test_dataset):
                print(f"Index phải từ 0 đến {len(test_dataset)-1}")
                continue
            
            # Load image
            img, true_label = test_dataset[idx]
            img_batch = img.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get prediction
            model.eval()
            with torch.no_grad():
                output = model(img_batch)
                probabilities = F.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                pred_label = predicted[0].cpu().item()
                confidence = probabilities[0][pred_label].cpu().item()
            
            # Display result
            print(f"\nImage {idx}:")
            print(f"True label: {true_label}")
            print(f"Predicted: {pred_label}")
            print(f"Confidence: {confidence:.3f}")
            
            # Show image
            plt.figure(figsize=(6, 6))
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f'True: {true_label}, Pred: {pred_label}, Conf: {confidence:.3f}')
            plt.axis('off')
            plt.show()
            
        except ValueError:
            print("Vui lòng nhập số hợp lệ")
        except KeyboardInterrupt:
            break
    
    print("Demo kết thúc!")


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Model path
    model_path = 'best_mamba_mnist.pth'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        print("Please train the model first using: python train_mamba.py")
        return
    
    # Load model
    print("Loading trained Mamba SSM model...")
    model = load_trained_model(model_path, device)
    print("Model loaded successfully!")
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Menu
    while True:
        print("\n" + "="*50)
        print("MAMBA SSM MNIST DEMO")
        print("="*50)
        print("1. Analyze Performance")
        print("2. Plot Confusion Matrix")
        print("3. Visualize Predictions")
        print("4. Interactive Demo")
        print("5. Exit")
        
        choice = input("\nChọn option (1-5): ").strip()
        
        if choice == '1':
            print("\nAnalyzing performance...")
            all_predictions, all_targets, all_probabilities, accuracy = analyze_performance(
                model, test_loader, device
            )
            
        elif choice == '2':
            print("\nPlotting confusion matrix...")
            all_predictions, all_targets, _, _ = analyze_performance(model, test_loader, device)
            plot_confusion_matrix(all_targets, all_predictions)
            
        elif choice == '3':
            print("\nVisualizing predictions...")
            visualize_predictions(model, test_loader, device)
            
        elif choice == '4':
            interactive_demo(model, device)
            
        elif choice == '5':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    import os
    main()
