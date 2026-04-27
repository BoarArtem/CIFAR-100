import torch
from torch.utils.data import DataLoader
import torchvision.datasets
import torchvision.transforms as transforms

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408,), (0.2675, 0.2565, 0.2761,))])

train_dataset = torchvision.datasets.CIFAR100(root='data/', train=True, transform=transforms, download=True)
test_dataset = torchvision.datasets.CIFAR100(root='data/', train=False, download=False)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)

for X_batch, y_batch in train_loader:
    print(X_batch.shape)
    print(y_batch.shape)

    break