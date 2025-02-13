import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau


train_transform = transforms.Compose([
    transforms.Resize((224, 126)),
    transforms.Pad((0, 49)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 126)),
    transforms.Pad((0, 49)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(
    'TRAIN_DIR_HERE',
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    'VAL_DIR_HERE',
    transform=val_transform
)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters(): 
    param.requires_grad = True

    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ])

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_val_accuracy = 0.0

    for epoch in range(15):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_correct += ((outputs > 0.5).float() == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_correct += ((outputs > 0.5).float() == labels).sum().item()

        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct / len(train_dataset)
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/15:')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}')
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), 
                    'PATH_TO_SAVE_MODEL_HERE/resnet18_nba_ad.pth')
            print('Model improved - saved new weights')
