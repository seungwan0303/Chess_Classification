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

def validate(model, valid_load, criterion):
    model.eval()

    valid_loss = 0.0
    valid_corrects = 0

    with torch.no_grad():
        for inputs, labels in valid_load:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)

            # TORCH.MAX가 뽑는 값을 이용하여 CHESS CLASSIFICATION 진행하기
            _, preds = torch.max(outputs, 1)
            valid_corrects += torch.sum(preds == labels.data)

    validation_loss = valid_loss / (len(valid_dataset) / batch_size)
    validation_accuracy = 100. * valid_corrects / len(valid_dataset)

    return validation_loss, validation_accuracy

def set_model_mode(mode):
    if mode == 'train':
        train(model, train_load, train_dataset, optimizer)
    elif mode == 'eval':
        valid(model, valid_load, criterion)
    else:
        raise ValueError("Invalid mode! Choose between 'train' and 'eval'.")

# 초기 모드는 학습 모드로 설정
mode = 'train'
epochs = 30
lr = 0.001
best_loss = 1000
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    user_input = input("Enter 't' for training mode, 'e' for evaluation mode, or 'q' to quit: ")

    for epoch in range(epochs):
        if user_input == 'train':
            if mode != 'train':
                mode = 'train'
                print(f'-------epoch {epoch + 1}')
                train_loss, train_accuracy = set_model_mode(mode)

            if best_loss > train_loss:
                best_model = model.state_dict()
                best_loss = train_loss

                print('train accuracy:', train_accuracy)
                print('train loss:', train_loss)

                torch.save(best_model, f'./{epochs}_{best_loss}.pth')

        elif user_input == 'eval':
            if mode != 'eval':
               mode = 'eval'
               print(f'-------epoch {epoch + 1}')
               valid_loss, valid_accuracy = set_model_mode(mode)

               if best_loss > valid_loss:
                  best_model = model.state_dict()
                  best_loss = valid_loss

               print('valid accuracy:', valid_accuracy)
               print('valid loss:', valid_loss)

        elif user_input == 'quit':
            print("Exiting...")
            break

        else:
            print("Invalid input! Please enter 'train', 'valid', or 'quit'.")
