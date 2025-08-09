import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from vision_mamba2 import create_vision_mamba2_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def load_trained_model(model_path, device):
    """
    Load model đã train
    """
    model = create_vision_mamba2_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_mnist_test_data():
    """
    Load MNIST test data
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    return test_dataset


def visualize_predictions(model, test_dataset, device, num_samples=16):
    """
    Visualize predictions của model
    """
    model.eval()
    
    # Lấy random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, true_label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            
            # Prediction
            output = model(image)
            prob = F.softmax(output, dim=1)
            predicted_label = output.argmax(dim=1).item()
            confidence = prob.max().item()
            
            # Plot
            axes[i].imshow(image.squeeze().cpu(), cmap='gray')
            color = 'green' if predicted_label == true_label else 'red'
            axes[i].set_title(f'Pred: {predicted_label}\nTrue: {true_label}\nConf: {confidence:.2f}')
            axes[i].set_xlabel(f'Confidence: {confidence:.2f}')
            axes[i].axis('off')
            
            if predicted_label == true_label:
                correct += 1
            total += 1
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Accuracy on visualized samples: {100 * correct / total:.2f}%")


def plot_confusion_matrix(model, test_dataset, device):
    """
    Vẽ confusion matrix
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, label = test_dataset[i]
            image = image.unsqueeze(0).to(device)
            
            output = model(image)
            prediction = output.argmax(dim=1).item()
            
            all_predictions.append(prediction)
            all_labels.append(label)
            
            if i % 1000 == 0:
                print(f"Processed {i}/{len(test_dataset)} samples")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - Vision Mamba 2 on MNIST')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))


def analyze_model_performance(model, test_dataset, device):
    """
    Phân tích performance của model
    """
    model.eval()
    
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, label = test_dataset[i]
            image = image.unsqueeze(0).to(device)
            
            output = model(image)
            prediction = output.argmax(dim=1).item()
            
            if prediction == label:
                correct += 1
                class_correct[label] += 1
            total += 1
            class_total[label] += 1
            
            if i % 1000 == 0:
                print(f"Processed {i}/{len(test_dataset)} samples")
    
    # Overall accuracy
    overall_accuracy = 100 * correct / total
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"Class {i}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return overall_accuracy


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = 'vision_mamba2_mnist.pth'
    try:
        model = load_trained_model(model_path, device)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    # Load test data
    test_dataset = get_mnist_test_data()
    print(f"Loaded {len(test_dataset)} test samples")
    
    # Analyze performance
    print("\n=== Model Performance Analysis ===")
    accuracy = analyze_model_performance(model, test_dataset, device)
    
    # Plot confusion matrix
    print("\n=== Confusion Matrix ===")
    plot_confusion_matrix(model, test_dataset, device)
    
    # Visualize predictions
    print("\n=== Prediction Visualization ===")
    visualize_predictions(model, test_dataset, device)
    
    print(f"\nAnalysis completed! Overall accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main() 