import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define the model
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
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Outputs a fixed 18x18 feature map
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),  # No change needed here
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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load data
    train_dataset = datasets.ImageFolder('C:/Users/weekly/PyCharmProjects/NBAadBlock/data/processed/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('C:/Users/weekly/PyCharmProjects/NBAadBlock/data/processed/validate', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=12)

    model = CNN().to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Function for training and evaluation
    def train(model, loader, optimizer, criterion, device, verbose=True):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if verbose:
                print(f'Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}')
        return running_loss / len(loader.dataset)

    def validate(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
        return running_loss / len(loader.dataset)

    # Training loop
    epochs = 10
    best_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'C:/Users/weekly/PyCharmProjects/NBAadBlock/models/best_nba_ad_model_128_image_size.pth')
