import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import os
from PIL import Image
import glob
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Tipos de DataAugmentation => left-shift / right-shift / geometria no geral
# Na validação, usar o k-fold para obter a melhor configuração (cross validation)


class ModelEvaluator:
    def __init__(self, model, classes):
        self.model = model
        self.classes = classes

    def evaluate(self, data_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calcula a matriz de confusão
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Relatório de classificação
        print(classification_report(all_labels, all_preds, target_names=self.classes))

        return conf_matrix

    def plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(12,10))
        sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


class LettersDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.files = [glob.glob(os.path.join(root_dir, cls, '*')) for cls in self.classes]
        self.files = [item for sublist in self.files for item in sublist]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]
        label = self.classes.index(os.path.basename(os.path.dirname(image_path)))
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = LettersDataset(root_dir=f'C:/Users/vitin/OneDrive/Área de Trabalho/TCC/crops', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


class LettersClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LettersClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*32*32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*32*32)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


model = LettersClassifier(num_classes=len(dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Loop de treinamento
print("Iniciando o treinamento...")
num_epochs = 100
for epoch in range(num_epochs):
    # Treinamento
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    # Validação
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {100.*correct/total:.2f}%")

# Salvar o modelo treinado
print("Salvando o modelo treinado...")
torch.save(model.state_dict(), 'letters_classifier_model.pt')
print("Modelo salvo como 'letters_classifier_model.pt'")

evaluator = ModelEvaluator(model, dataset.classes)
conf_matrix = evaluator.evaluate(val_loader)
evaluator.plot_confusion_matrix(conf_matrix)
