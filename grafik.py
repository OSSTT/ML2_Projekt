import json
import matplotlib.pyplot as plt

# JSON-Dateien laden
with open(r'results\trained18_results.json', 'r') as f:
    results_18 = json.load(f)

with open(r'results\trained50_results.json', 'r') as f:
    results_50 = json.load(f)

# Daten extrahieren
epochs_18 = [entry['epoch'] for entry in results_18 if 'epoch' in entry]
loss_18 = [entry['loss'] for entry in results_18 if 'epoch' in entry]
val_loss_18 = [entry['val_loss'] for entry in results_18 if 'epoch' in entry]
val_acc_18 = [entry['val_acc'] for entry in results_18 if 'epoch' in entry]

final_train_loss_18 = results_18[-1]['final_train_loss']
final_val_loss_18 = results_18[-1]['final_val_loss']
final_val_acc_18 = results_18[-1]['final_val_acc']
test_loss_18 = results_18[-1]['test_loss']
test_accuracy_18 = results_18[-1]['test_accuracy']

epochs_50 = [entry['epoch'] for entry in results_50 if 'epoch' in entry]
loss_50 = [entry['loss'] for entry in results_50 if 'epoch' in entry]
val_loss_50 = [entry['val_loss'] for entry in results_50 if 'epoch' in entry]
val_acc_50 = [entry['val_acc'] for entry in results_50 if 'epoch' in entry]

final_train_loss_50 = results_50[-1]['final_train_loss']
final_val_loss_50 = results_50[-1]['final_val_loss']
final_val_acc_50 = results_50[-1]['final_val_acc']
test_loss_50 = results_50[-1]['test_loss']
test_accuracy_50 = results_50[-1]['test_accuracy']

# Grafiken erstellen
plt.figure(figsize=(18, 6))

# Validierungsverlust
plt.subplot(1, 3, 1)
plt.plot(epochs_18, val_loss_18, label='ResNet18')
plt.plot(epochs_50, val_loss_50, label='ResNet50')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss over Epochs')
plt.legend()
plt.xticks(range(min(epochs_18 + epochs_50), max(epochs_18 + epochs_50) + 1))  # X-Achse auf ganze Zahlen setzen

# Validierungsgenauigkeit
plt.subplot(1, 3, 2)
plt.plot(epochs_18, val_acc_18, label='ResNet18')
plt.plot(epochs_50, val_acc_50, label='ResNet50')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.legend()
plt.xticks(range(min(epochs_18 + epochs_50), max(epochs_18 + epochs_50) + 1))  # X-Achse auf ganze Zahlen setzen

# Testverlust und Testgenauigkeit
plt.subplot(1, 3, 3)
bar_width = 0.35
index = [0, 1]

test_metrics_18 = [test_loss_18, test_accuracy_18]
test_metrics_50 = [test_loss_50, test_accuracy_50]

bar1 = plt.bar(index, test_metrics_18, bar_width, label='ResNet18')
bar2 = plt.bar([i + bar_width for i in index], test_metrics_50, bar_width, label='ResNet50')

plt.xlabel('Metric')
plt.ylabel('Value')
plt.title('Final Results')
plt.xticks([i + bar_width / 2 for i in index], ['Loss', 'Accuracy'])
plt.legend()

# Annotate bars with their values
for bar in bar1 + bar2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()

# Grafik speichern
output_path = 'results/training_data.jpg'
plt.savefig(output_path, format='jpg')

# Grafik anzeigen
plt.show()
