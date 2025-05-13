# **
# Model's training, using the lists created by "create:dataset.py"
# Saves all the best weights into the folder "weights" during the training
# **

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import os
import random
from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn import metrics

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#load dataset
class MySongs(Dataset):
  def __init__(self, songs, labels):
    self.songs = songs
    self.labels = labels
  def __len__(self):
    return len(self.songs)
  def __getitem__(self, idx):
    return self.songs[idx], self.labels[idx]



def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, out_path): #change_lr=None
  min_valid_loss = 10**10
  for epoch in tqdm(range(1,epochs+1)):
    model.train()
    batch_losses=[]
    '''if change_lr:
      optimizer = change_lr(optimizer, epoch)'''
    for i, data in enumerate(train_loader):
      x, y = data
      optimizer.zero_grad()
      x = x.to(device, dtype=torch.float32)
      y = y.to(device, dtype=torch.long)
      x = x.unsqueeze(1)
      y_hat = model(x)
      loss = loss_fn(y_hat, y)
      loss.backward()
      batch_losses.append(loss.item())
      optimizer.step()

    train_losses.append(batch_losses)
    print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
    model.eval()
    batch_losses=[]
    trace_y = []
    trace_yhat = []
    for i, data in enumerate(valid_loader):
      x, y = data
      x = x.to(device, dtype=torch.float32)
      y = y.to(device, dtype=torch.long)
      x = x.unsqueeze(1)
      y_hat = model(x)
      loss = loss_fn(y_hat, y)
      trace_y.append(y.cpu().detach().numpy())
      trace_yhat.append(y_hat.cpu().detach().numpy())
      batch_losses.append(loss.item())

    valid_losses.append(batch_losses)
    trace_y = np.concatenate(trace_y)
    trace_yhat = np.concatenate(trace_yhat)
    accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
    print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')
    if np.mean(valid_losses[-1]) < min_valid_loss:
          min_valid_loss = np.mean(valid_losses[-1])
          #save the whole model
          torch.save(model, out_path + 'BBmodel_e' + str(epoch) + '_acc' + str(round(accuracy, 2)) +'.pt')
          ''' #save only weights
          torch.save(model.state_dict(), out_path + 'weights_BBmodel' + str(epoch) + '_acc' + str(round(accuracy, 2)) +'.pt')'''


  return model





#LOAD DATASET
train_songs = torch.load('lists/train_mels.pt')
train_classes = torch.load('lists/train_classes.pt')

valid_songs = torch.load('lists/valid_mels.pt')
valid_classes = torch.load('lists/valid_classes.pt')


#DATALOADER
train_data = MySongs(train_songs.numpy(), train_classes.numpy())
valid_data = MySongs(valid_songs.numpy(), valid_classes.numpy())
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)


#BUILD MODEL
resnet_model = resnet34(weights=ResNet34_Weights.DEFAULT)
resnet_model.fc = nn.Linear(512,2) #change to output 2 classes instead of 1000
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #change to accept from 3 to one channel
resnet_model = resnet_model.to(device)


#TRAINING
learning_rate = 2e-4
optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate)
epochs = 50
loss_fn = nn.CrossEntropyLoss()
resnet_train_losses=[]
resnet_valid_losses=[]
out_path = 'weights/'
if not os.path.exists(out_path):
    os.mkdir(out_path)

model = train(resnet_model, loss_fn, train_loader, valid_loader, epochs, optimizer, resnet_train_losses, resnet_valid_losses, out_path)
