import sys

import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from dataload import *
from torchvision import transforms

import time
import copy
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 변수 선언부

file_paths = 'Chessman-image-dataset'
train_paths = 'Chessman-image-dataset/train'
valid_paths = 'Chessman-image-dataset/valid'
test_paths = 'Chessman-image-dataset/test'
batch_size = 16

transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

train_data = torchvision.datasets.ImageFolder(root=train_paths)
valid_data = torchvision.datasets.ImageFolder(root=valid_paths)
test_data = torchvision.datasets.ImageFolder(root=test_paths)

train_dataset = CustomDataset(train_data, transforms)
valid_dataset = CustomDataset(valid_data,transforms)

train_load = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
valid_load = DataLoader(valid_dataset,batch_size=batch_size, shuffle=False, num_workers=5)

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
        loss = criterion(outputs, label)

        _, preds = torch.max(outputs.data, 1)

        train_running_loss += loss.item()
        train_running_correct += (preds == label).sum().item()

        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / (len(train_dataset) / batch_size)
    train_accuracy = 100. * train_running_correct / len(train_dataset)

    return train_loss, train_accuracy

def validate(model, valid_load, valid_dataset, criterion):
    model.eval()

    validation_loss = 0.0
    validation_corrects = 0

    pbar = tqdm.tqdm(valid_load, unit='batch')

    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

            # TORCH.MAX가 뽑는 값을 이용하여 CHESS CLASSIFICATION 진행하기
            _, preds = torch.max(outputs, 1)

            validation_loss += (preds == labels).sum().item()
            validation_corrects += (preds == labels).sum().item()

        valid_loss = validation_loss / (len(valid_dataset) / batch_size)
        valid_accuracy = 100. * validation_corrects / len(valid_dataset)

    return valid_loss, valid_accuracy

def set_model_mode(mode):
    if mode == 'train':
        return train(model, train_load, train_dataset, optimizer)
    elif mode == 'eval':
        return validate(model, valid_load, valid_dataset, criterion)
    else:
        raise ValueError("Invalid mode! Choose between 'train' and 'eval'.")

if __name__ == '__main__':
    # 초기 모드 설정
    mode = input("Enter 'train' for training mode, 'eval' for evaluation mode: ")
    key = mode

    epochs = 30
    lr = 0.001
    best_loss = 1000
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if key == 'train':
        for epoch in range(epochs):
            print(f'-------epoch {epoch + 1}')

            train_loss, train_accuracy = set_model_mode(key)

            if best_loss > train_loss:
                best_model = model.state_dict()
                best_loss = train_loss

            print('train accuracy:', train_accuracy)
            print('train loss:', train_loss)

            torch.save(best_model, f'./{epochs}_{best_loss}.pth')

    elif key == 'eval':
        for epoch in range(epochs):
            print(f'-------epoch {epoch + 1}')

            valid_loss, valid_accuracy = set_model_mode(key)

            if best_loss > valid_loss:
                best_model = model.state_dict()
                best_loss = valid_loss

            print('valid accuracy:', valid_accuracy)
            print('valid loss:', valid_loss)

            torch.save(best_model, f'./{epochs}_{best_loss}.pth')

    elif key == 'quit':
        print("Exiting...")

    else:
        print("Invalid input! Please enter 'train', 'valid', or 'quit'.")
