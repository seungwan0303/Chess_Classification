from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from dataload import *

import time
import copy
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

file_paths = 'Chessman-image-dataset'
train_paths = 'Chessman-image-dataset/train'
valid_paths = 'Chessman-image-dataset/valid'
test_paths = 'Chessman-image-dataset/test'
batch_size=16

transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

train_data = torchvision.datasets.ImageFolder(root=train_paths)
valid_data = torchvision.datasets.ImageFolder(root=valid_paths)
test_data = torchvision.datasets.ImageFolder(root=test_paths)

train_dataset = CustomDataset(train_data, transforms)

# DataLoad = {
#              'Train' : DataLoader(CustomDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=5),
#              'Test' : DataLoader(CustomDataset(test_data),batch_size=batch_size, shuffle=False, num_workers=5),
#              'Valid' : DataLoader(CustomDataset(valid_data),batch_size=batch_size, shuffle=False, num_workers=5)
# }

train_load = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

def resnet50():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 6)
    return model

device_txt = 'cuda:0'
device = torch.device(device_txt)
model = resnet50().to(device)

def train(model, train_load, train_dataset, optimizer):
    model.train()

    train_running_loss = 0.0
    train_running_correct = 0

    pbar = tqdm.tqdm(train_load, unit='batch')
    for img, label in pbar:
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        outputs = model(img)
        print(outputs.data)
        print(outputs.data.shape)
        # sys.exit()

        loss = criterion(outputs, label)
        train_running_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)
        # print('preds', preds)
        # print('labels:', label)

        train_running_correct += (preds == label).sum().item()

        loss.backward()
        optimizer.step()

    _train_loss = train_running_loss / (len(train_dataset) / batch_size)
    _train_accuracy = 100. * train_running_correct / len(train_dataset)

    return _train_loss, _train_accuracy

def validate():
    pass

epochs = 20
lr = 0.001
best_loss = 1000
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    for epoch in range(epochs):
        print(f'-------epoch {epoch + 1}')

        phase = ['train', 'valid'] if (epoch + 1) / 10 == 0 else ['train']

        train_loss, train_accuracy = train(model, train_load, train_dataset, optimizer)

        if best_loss > train_loss:
            best_model = model.state_dict()
            best_loss = train_loss

        print('train accuracy:', train_accuracy)
        print('train loss:', train_loss)

    torch.save(best_model, f'./{epochs}_{best_loss}.pth')

