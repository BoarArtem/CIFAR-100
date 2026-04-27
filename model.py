import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CIFAR100Model(nn.Module):
    def __init__(self):
        super(CIFAR100Model, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 100)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)

        return x

model = CIFAR100Model().to(device)

criterion = nn.CrossEntropyLoss()
optimization = optim.Adam(model.parameters(), lr=0.001)

def train(epochs):
    print("It successfully starting...")
    for epoch in range(epochs):
        training_loss = 0
        correct = 0
        total = 0

        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimization.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimization.step()

            training_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total

        print(f"Epochs: {epoch+1}/{epochs}, Loss: {training_loss / len(train_loader):.4f}, Accuracy: {int(accuracy*100)}%")
        torch.save(model.state_dict(), "cifar100_model.pth")

if __name__ == "__main__":
    print(train(40))