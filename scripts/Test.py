import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

test_transform = transforms.Compose([
    transforms.Resize((224, 126)),
    transforms.Pad((0, 49)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('PATH_TO_MODEL_HERE/resnet18_nba_ad.pth')
    
    test_dataset = datasets.ImageFolder(
        'TEST_DIR_HERE',
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(inputs)
            correct += ((outputs > 0.5).float() == labels).sum().item()
    
    print(f'Test Accuracy: {correct / len(test_dataset):.2%}')