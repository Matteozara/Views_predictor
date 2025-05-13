# **
# Model test, using the lists created by "create:dataset.py"
# Change the name of the best weight iside the "weights" folder in "best.pt"
# **

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import random
from torchvision.models import resnet34
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


def evaluation(model, valid_loader): #change_lr=None
  loss_tot = 0
  loss_fn = nn.CrossEntropyLoss()
  predicted = []
  gt = []
  for i, data in enumerate(valid_loader):
    print("iteration: ", i)
    x, y = data
    x = x.to(device, dtype=torch.float32)
    y = y.to(device, dtype=torch.long)
    x = x.unsqueeze(1)
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    gt.append(y.cpu().detach().numpy())
    predicted.append(y_hat.cpu().detach().numpy())
    loss_tot += loss.item()

  gt_np = np.concatenate(gt) #convert in one numpy list
  pred_np = np.concatenate(predicted)
  pred_np = pred_np.argmax(axis=1) #take the prediction (max probability) for each song
  print("pred_np: ", pred_np)
  print("gt_np: ", gt_np)
  accuracy = np.mean(pred_np==gt_np)
  correct = 0
  tot_less = 0
  tot_more = 0
  pred_less = 0
  pred_more = 0
  for i in range(0, len(gt_np)):
    if gt_np[i] == pred_np[i]:
      correct += 1

    if gt_np[i] == 0:
      tot_less += 1
    else:
      tot_more += 1

    if pred_np[i] == 0:
      pred_less += 1
    else:
      pred_more += 1
  print("Songs corrected classified: ", correct, " on a total of ", len(gt_np))
  print("Accuracy: ", accuracy)
  print("total loss: ", loss_tot, " and loss per song: ", (loss_tot/len(gt_np)))
  print("Tot predicted <300M views: ", pred_less, "on a total of <300M soongs: ", tot_less)
  print("Tot predicted >300M views: ", pred_more, "on a total of >300M soongs: ", tot_more)

  return pred_np, gt_np




#LOAD BEST MODEL
model_path = "weights/best.pt"
model = torch.load(model_path, map_location=device,  weights_only=False)
#model = model.to(device)
model.eval()


#LOAD DATA
valid_songs = torch.load('lists/test_mels.pt') #torch.load('data/valid_mels.pt')
valid_classes = torch.load('lists/test_classes.pt') #torch.load('data/valid_classes.pt')

valid_data = MySongs(valid_songs.numpy(), valid_classes.numpy())
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)



#PREDICTION
predicted, gt = evaluation(model, valid_loader)
# 0 = <300M
# 1 = >300M



#CONFUSION MATRIX
confusion_matrix = metrics.confusion_matrix(gt, predicted)#, normalize = 'pred')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels = ['-300M', '+300M'])

fig, ax = plt.subplots(figsize=(10, 10))
plt.rcParams.update({'font.size': 32})
cm_display.plot(ax=ax)
fig.savefig('res/confusion_matrix.png', bbox_inches='tight', dpi=300)
plt.figure()