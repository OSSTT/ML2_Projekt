import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import json
from datetime import datetime

# Definition der Klasse-Namen basierend auf den Ordnern im Datenset
data_dir = 'datenset/'
class_names = sorted(os.listdir(data_dir))

# Definition des CustomResNet-Modells
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

# Laden des Basismodells
base_model = models.resnet18(weights=None)
num_classes = len(class_names)
model = CustomResNet(base_model, num_classes)
model.load_state_dict(torch.load('trained18_model.pth'))
model.eval()

# Vorverarbeitung der Eingabe
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

results_dir = 'results'
results_file = os.path.join(results_dir, 'prediction_results.json')

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def predict_image(image_path):
    # Laden und Vorverarbeiten des Testbildes
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Hinzufügen einer zusätzlichen Dimension für den Batch

    # Vorhersage machen
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Ergebnis ausgeben
    predicted_class = class_names[predicted.item()]
    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'predicted_class': predicted_class
    }

    # JSON-Datei lesen und aktualisieren
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = []

    results.append(result)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    return predicted_class
