import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import time
import copy


ground_truth_list = []
answer_list = []
total_epoch = 1
Leaning_Rate = 0.001
'''
total_epoch = 1000
Leaning_Rate = 0.001
'''
device = torch.device("cpu")

TRAIN_PATH = "./data/bird_train"
TEST_PATH = "./data/bird_test"

simple_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = ImageFolder(TRAIN_PATH, simple_transform)
testset = ImageFolder(TEST_PATH, simple_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=4, shuffle=False)
dataloaders = {
    "train": train_loader,
    "test": test_loader
}
datasizes = {
    "train": len(trainset),
    "test": len(testset)
}
CLASSES = list(trainset.class_to_idx.keys())

train_losses = []
train_accuracy = []
val_losses = []
val_accuracy = []


class Net(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 2809)
        self.fc2 = nn.Linear(2809, 512)
        self.fc3 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, -1)




def train_model(model, criterion, optimizer, scheduler, epochs=total_epoch):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print("training:  Epoch {}/{}".format(epoch + 1, total_epoch))
        print("-" * 10)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / datasizes[phase]
            epoch_acc = running_corrects.double() / datasizes[phase]


            if (phase == "train"):
                a = epoch_loss
                b = epoch_acc
                train_losses.append(a)
                train_accuracy.append(b)

            if (phase == "test"):
                c = epoch_loss
                d = epoch_acc
                val_losses.append(c)
                val_accuracy.append(d)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            if (phase == "test" and epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())





        print()

    time_elapsed = time.time() - since
    print("Training complete in {:0f}m {:0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {}".format(best_acc))

    x_len = np.arange(len(train_losses))
    plt.plot(x_len, train_losses, marker='.', lw=1, c='red', label="train_losses")
    plt.plot(x_len, val_losses, marker='.', lw=1, c='cyan', label="val_losses")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.plot(x_len, val_accuracy, marker='.', lw=1, c='green', label="val_accuracy")
    plt.plot(x_len, train_accuracy, marker='.', lw=1, c='blue', label="train_accuracy")

    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    model.load_state_dict(best_model_wts)
    return model



if __name__ == '__main__':
    model_ft = models.resnet18(pretrained=True)

    for param in model_ft.parameters():
        param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(CLASSES))
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.003, momentum=0.9)
    exp_lr_sc = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_sc, epochs=total_epoch)




