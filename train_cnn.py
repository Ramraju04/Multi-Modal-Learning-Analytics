import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vision_model import EngagementCNN
import os

# Configuration
DATA_DIR = 'data/Student-engagement-dataset/Engaged' # Using subfolder structure
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5 # Quick training for demonstration
LEARNING_RATE = 0.001

def train():
    print("Initializing Training Pipeline...")
    
    # 1. Data Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Dataset
    # Structure: data/Student-engagement-dataset/Engaged/{confused, engaged, frustrated}
    print(f"Loading data from {DATA_DIR}...")
    try:
        dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
        print(f"Classes found: {dataset.classes}")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error loading data: {e}. Please ensure folder structure matches PyTorch ImageFolder requirements.")
        return

    # 3. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = EngagementCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(dataloader):.4f} | Accuracy: {epoch_acc:.2f}%")

    print('Finished Training')

    # 5. Save Model
    torch.save(model.state_dict(), 'engagement_model.pth')
    print("Model saved to 'engagement_model.pth'")
    
    # Save Class Mapping
    with open('class_map.txt', 'w') as f:
        for idx, cls in enumerate(dataset.classes):
            f.write(f"{idx}:{cls}\n")

if __name__ == '__main__':
    train()
