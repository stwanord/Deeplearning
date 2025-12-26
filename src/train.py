import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import FireDetectionCNN
from dataset import get_data_loaders
import os

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size)
    
    if train_loader is None:
        print("Dataset not found or empty.")
        return

    num_classes = len(class_names)
    print(f"Initializing model for {num_classes} classes: {class_names}")

    model = FireDetectionCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_accuracies = []

    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
     
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Val Accuracy: {val_acc:.2f}%")


    torch.save(model.state_dict(), "fire_model.pth")
    print("Model saved as fire_model.pth")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    
    plt.savefig('training_results.png')
    print("Training graphs saved as training_results.png")

if __name__ == "__main__":
  
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if os.path.exists(DATA_DIR):
        train_model(DATA_DIR)
    else:
        print(f"Please create 'data' folder at {DATA_DIR} and put your dataset inside.")
