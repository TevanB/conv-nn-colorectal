import os
import random
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import ColorectalCancerDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data_dir = "./data"
# Set seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define data augmentation transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),  # Add this line
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),  # Add this line
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Get the list of classes and their corresponding directories
classes = os.listdir(data_dir)
class_dirs = [os.path.join(data_dir, class_name) for class_name in classes]

# Load the image paths and corresponding labels
img_paths, labels = [], []
for label, class_dir in enumerate(class_dirs):
    for img_name in os.listdir(class_dir):
        img_paths.append(os.path.join(class_dir, img_name))
        labels.append(label)

# Split the data into train, validation, and test sets (70% train, 15% validation, 15% test)
train_img_paths, temp_img_paths, train_labels, temp_labels = train_test_split(
    img_paths, labels, test_size=0.3, stratify=labels, random_state=42
)
val_img_paths, test_img_paths, val_labels, test_labels = train_test_split(
    temp_img_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

train_dataset = ColorectalCancerDataset(train_img_paths, train_labels, transform=train_transform)
val_dataset = ColorectalCancerDataset(val_img_paths, val_labels, transform=test_transform)
test_dataset = ColorectalCancerDataset(test_img_paths, test_labels, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

densenet = models.densenet121(pretrained=True)
num_classes = len(set(train_labels))
densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
densenet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(densenet.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

num_epochs = 50

accuracies = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Train the model
    densenet.train()
    train_loss = 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = densenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Train Loss: {train_loss}")

    # Validate the model
    densenet.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = densenet(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total
    accuracies.append(val_acc)
    if val_acc > max(accuracies):
        torch.save(densenet.state_dict(), "best_model.pt")
    print(f"Val Loss: {val_loss}, Val Acc: {val_acc}")

print("Average accuracy: ", sum(accuracies) / len(accuracies))


# Test the model
densenet.load_state_dict(torch.load("best_model.pt"))  # Load the best model
densenet.eval()

all_outputs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = densenet(inputs)
        _, predicted = outputs.max(1)
        all_outputs.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(all_labels, all_outputs)
test_precision = precision_score(all_labels, all_outputs, average='weighted')
test_recall = recall_score(all_labels, all_outputs, average='weighted')
test_f1 = f1_score(all_labels, all_outputs, average='weighted')

print("Test Accuracy: {:.4f}".format(test_accuracy))
print("Test Precision: {:.4f}".format(test_precision))
print("Test Recall: {:.4f}".format(test_recall))
print("Test F1-score: {:.4f}".format(test_f1))
