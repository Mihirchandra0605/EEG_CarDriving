# cnn_model.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report
import torch.nn.functional as F

# Load and normalize data
X = np.load("../src/X_full_nolda.npy")
y = np.load("../src/y_full_nolda.npy")
X = (X - X.mean()) / X.std()

# Reshape for Conv1D: (samples, channels=1, features)
X = X.reshape((X.shape[0], 1, 336))

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Dataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# Model
class RawCNN(nn.Module):
    def __init__(self):
        super(RawCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 84, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (batch, 16, 168)
        x = self.pool(F.relu(self.conv2(x)))   # (batch, 32, 84)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)              # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RawCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
print(" Training CNN...")
for epoch in range(30):
    model.train()
    loss_val = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
    print(f"Epoch [{epoch+1}/30], Loss: {loss_val/len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n Test Accuracy:", np.mean(np.array(all_preds) == np.array(all_labels)))
print("\n Classification Report:")
print(classification_report(all_labels, all_preds))
