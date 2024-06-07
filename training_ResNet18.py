import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import json
import os
import copy

# Daten laden und vorverarbeiten
data_dir = 'datenset/'
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Aufteilung in Trainings-, Validierungs- und Testdaten
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Custom ResNet-Modell mit Dropout definieren
class CustomResNet(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomResNet, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)
        self.base_model.fc = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Basismodell laden und anpassen
base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = CustomResNet(base_model, len(dataset.classes))

# Optimizer, Loss und Scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# Early Stopping Klasse
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

early_stopping = EarlyStopping(patience=7)

# Training und Validierung
num_epochs = 32
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
training_results = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_dataset)
    val_acc = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Early Stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    # Adjust learning rate
    scheduler.step(val_loss)
    
    training_results.append({
        'epoch': epoch + 1,
        'loss': epoch_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'batch_size': 32,
        'lr': scheduler.optimizer.param_groups[0]['lr']
    })

# Laden der besten Gewichte
model.load_state_dict(best_model_wts)

# Modell evaluieren
correct = 0
total = 0
test_loss = 0.0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss /= len(test_dataset)
accuracy = correct / total
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Final results
final_results = {
    'final_train_loss': epoch_loss,
    'final_val_loss': val_loss,
    'final_val_acc': val_acc,
    'test_loss': test_loss,
    'test_accuracy': accuracy
}

training_results.append(final_results)

# Ergebnisse als JSON speichern
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

results_file = os.path.join(results_dir, 'trained18_results.json')

with open(results_file, 'w') as f:
    json.dump(training_results, f, indent=4)

# Modell speichern
torch.save(model.state_dict(), 'trained18_model.pth')
