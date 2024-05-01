import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

# Define the CNN model architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((18, 18))  # Outputs a fixed 18x18 feature map
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 512),  # No change needed here
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.adaptive_pool(x)  # Apply adaptive pooling
        x = self.fc_layer(x)
        return x
def load_model(model_path):
    # Initialize the model
    model = CNN()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model

def predict(model, loader):
    results = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            # Convert outputs to binary predictions by rounding and convert to integers
            predicted_classes = outputs.round().int().squeeze().tolist()  # Adjust here to ensure proper format
            labels = labels.tolist()
            results.extend(zip(predicted_classes, labels))
    return results

if __name__ == '__main__':
    model_path = 'C:/Users/weekly/PyCharmProjects/NBAadBlock/models/best_nba_ad_model_256_image_size.pth'
    test_dir = 'C:/Users/weekly/PyCharmProjects/NBAadBlock/data/processed/test'  # Test directory path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path).to(device)

    # Define the transformations and DataLoader for the test set
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    predictions = predict(model, test_loader)
    correct = sum(1 for pred, label in predictions if pred == label)
    accuracy = correct / len(predictions)
    print(f'Accuracy: {accuracy:.2f}')
