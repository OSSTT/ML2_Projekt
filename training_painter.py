import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json
import os

# Daten laden und vorverarbeiten
data_dir = 'datenset/'
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Aufteilung in Trainings- und Testdaten
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Modell definieren (ResNet18 mit Anpassung des Ausgabeknotens)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))

# Modell trainieren
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
num_epochs = 32

training_results = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    training_results.append({
        'epoch': epoch + 1,
        'loss': epoch_loss,
        'batch_size': batch_size
    })

# Modell evaluieren
test_loader = DataLoader(test_dataset, batch_size=batch_size)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')
training_results.append({
    'accuracy': accuracy
})

# Ergebnisse als JSON speichern
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

results_file = os.path.join(results_dir, 'training_results.json')

with open(results_file, 'w') as f:
    json.dump(training_results, f, indent=4)

# Modell speichern
torch.save(model.state_dict(), 'trained_model.pth')
