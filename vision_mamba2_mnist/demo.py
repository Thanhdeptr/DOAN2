import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from vision_simple import create_vision_simple_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def load_trained_model(model_path, device):
    """
    Load model đã train
    """
    model = create_vision_simple_model().to(device)
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
            axes[i].set_title(f'Pred: {predicted_label}\nTrue: {true_label}\nConf: {confidence:.2f}', 
                            color=color, fontsize=10)
            axes[i].axis('off')
            
            if predicted_label == true_label:
                correct += 1
            total += 1
    
    plt.tight_layout()
    plt.savefig('demo_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Accuracy on demo samples: {100 * correct / total:.2f}%")


def plot_confusion_matrix(model, test_dataset, device, num_samples=1000):
    """
    Vẽ confusion matrix
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            image, label = test_dataset[i]
            image = image.unsqueeze(0).to(device)
            
            output = model(image)
            prediction = output.argmax(dim=1).item()
            
            all_predictions.append(prediction)
            all_labels.append(label)
            
            if i % 100 == 0:
                print(f"Processed {i}/{num_samples} samples")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - Vision Simple on MNIST')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('demo_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))


def analyze_model_performance(model, test_dataset, device, num_samples=1000):
    """
    Phân tích performance của model
    """
    model.eval()
    
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            image, label = test_dataset[i]
            image = image.unsqueeze(0).to(device)
            
            output = model(image)
            prediction = output.argmax(dim=1).item()
            
            if prediction == label:
                correct += 1
                class_correct[label] += 1
            total += 1
            class_total[label] += 1
            
            if i % 100 == 0:
                print(f"Processed {i}/{num_samples} samples")
    
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


def interactive_demo(model, test_dataset, device):
    """
    Demo tương tác - test từng ảnh một
    """
    model.eval()
    
    print("\n=== Interactive Demo ===")
    print("Nhập số từ 0-9999 để test ảnh tương ứng (hoặc 'q' để thoát):")
    
    while True:
        try:
            user_input = input("Nhập index (0-9999): ").strip()
            
            if user_input.lower() == 'q':
                break
            
            idx = int(user_input)
            if idx < 0 or idx >= len(test_dataset):
                print(f"Index phải từ 0 đến {len(test_dataset)-1}")
                continue
            
            # Load image
            image, true_label = test_dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            
            # Prediction
            with torch.no_grad():
                output = model(image_tensor)
                prob = F.softmax(output, dim=1)
                predicted_label = output.argmax(dim=1).item()
                confidence = prob.max().item()
            
            # Display
            plt.figure(figsize=(8, 4))
            
            plt.subplot(1, 2, 1)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f'Image {idx}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            # Plot confidence for all classes
            probs = prob.squeeze().cpu().numpy()
            plt.bar(range(10), probs)
            plt.title(f'Predictions\nTrue: {true_label}, Pred: {predicted_label}\nConfidence: {confidence:.2f}')
            plt.xlabel('Digit')
            plt.ylabel('Probability')
            plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.show()
            
            # Print result
            result = "✅ CORRECT" if predicted_label == true_label else "❌ WRONG"
            print(f"Image {idx}: True={true_label}, Predicted={predicted_label}, Confidence={confidence:.2f} {result}")
            
        except ValueError:
            print("Vui lòng nhập số hợp lệ")
        except KeyboardInterrupt:
            break
    
    print("Demo kết thúc!")


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = 'vision_simple_mnist.pth'
    try:
        model = load_trained_model(model_path, device)
        print(f"✅ Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"❌ Model file {model_path} not found. Please train the model first.")
        return
    
    # Load test data
    test_dataset = get_mnist_test_data()
    print(f"✅ Loaded {len(test_dataset)} test samples")
    
    print("\n=== Vision Simple MNIST Demo ===")
    print("1. Analyze model performance")
    print("2. Visualize predictions")
    print("3. Plot confusion matrix")
    print("4. Interactive demo")
    print("5. Run all demos")
    
    choice = input("\nChọn option (1-5): ").strip()
    
    if choice == '1':
        print("\n=== Model Performance Analysis ===")
        accuracy = analyze_model_performance(model, test_dataset, device)
        
    elif choice == '2':
        print("\n=== Prediction Visualization ===")
        visualize_predictions(model, test_dataset, device)
        
    elif choice == '3':
        print("\n=== Confusion Matrix ===")
        plot_confusion_matrix(model, test_dataset, device)
        
    elif choice == '4':
        print("\n=== Interactive Demo ===")
        interactive_demo(model, test_dataset, device)
        
    elif choice == '5':
        print("\n=== Running All Demos ===")
        
        print("\n1. Performance Analysis:")
        accuracy = analyze_model_performance(model, test_dataset, device)
        
        print("\n2. Prediction Visualization:")
        visualize_predictions(model, test_dataset, device)
        
        print("\n3. Confusion Matrix:")
        plot_confusion_matrix(model, test_dataset, device)
        
        print("\n4. Interactive Demo:")
        interactive_demo(model, test_dataset, device)
        
    else:
        print("Invalid choice!")
    
    print(f"\n🎉 Demo completed! Model accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
