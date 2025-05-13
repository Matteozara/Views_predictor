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



def prediction_song(model, valid_loader):
    predicted = []
    model.eval()
    with torch.no_grad():  # disable gradients for evaluation
        for i, (x, _) in enumerate(valid_loader):
            print("iteration: ", i)
            x = x.to(device, dtype=torch.float32)
            x = x.unsqueeze(1)
            y_hat = model(x)
            print("y_hat: ", y_hat)
            predicted.append(y_hat.cpu().numpy())
    pred_np = np.concatenate(predicted)
    pred_np = pred_np.argmax(axis=1)
    return pred_np





#LOAD SONG
test_songs = torch.load('data/test_mels.pt')
test_classes = torch.load('data/test_classes.pt')

test_data = MySongs(test_songs.numpy(), test_classes.numpy())
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

#LOAD BEST MODEL
model_path = "weights/best/best_4songs_15sec.pt"
model = torch.load(model_path, map_location=device,  weights_only=False)
#model = model.to(device)
model.eval()


#PREDICTION
predicted = prediction_song(model, test_loader)
print("Output: ", predicted)
if predicted.sum() >= 2:
  print("Predicted > 300M (1)")
else:
  print("Predicted < 300M (0)")