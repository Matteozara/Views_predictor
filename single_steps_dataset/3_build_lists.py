import json
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split


path_root = "dataset/"
path_out = "lists/"
lista_songs = []
lista_class = []
lista_labels = []
flag = 0

with open("views.json", 'r') as json_file:
    data = json.load(json_file)
    for i in data:
        mel_spectrograms = np.load(path_root + i + '.npy')
        if data[i] < 300000000:
            flag = 0
        else: 
            flag = 1
        for mel in mel_spectrograms:
            lista_songs.append(mel)
            lista_labels.append(data[i])
            lista_class.append(flag)

print("### END ###")
print("length mels list: ", len(lista_songs))
print("length labels list: ", len(lista_labels))
print("length all list: ", len(lista_class))

#split train/valid

valid_size = 0.06 #10%

lista_songs = np.array(lista_songs)
lista_labels = np.array(lista_labels)
lista_class = np.array(lista_class)

train_indices, valid_indices = train_test_split(range(len(lista_songs)), test_size=valid_size, random_state=42)

lista_songs_train = lista_songs[train_indices]
lista_labels_train = lista_labels[train_indices]
lista_class_train = lista_class[train_indices]

lista_songs_valid = lista_songs[valid_indices]
lista_labels_valid = lista_labels[valid_indices]
lista_class_valid = lista_class[valid_indices]

train_songs = torch.tensor(lista_songs_train)
train_labels = torch.tensor(lista_labels_train)
train_classes = torch.tensor(lista_class_train)

valid_songs = torch.tensor(lista_songs_valid)
valid_labels = torch.tensor(lista_labels_valid)
valid_classes = torch.tensor(lista_class_valid)


print("TRAIN")
print("shape mels list: ", train_songs.shape)
print("shape labels list: ", train_labels.shape)
print("shape labels list: ", train_classes.shape)

print("VALID")
print("shape mels list: ", valid_songs.shape)
print("shape labels list: ", valid_labels.shape)
print("shape labels list: ", valid_classes.shape)


torch.save(train_songs, path_out +"train_mels.pt")
torch.save(train_labels, path_out + "train_labels.pt")
torch.save(train_classes, path_out +"train_classes.pt")

torch.save(valid_songs, path_out +"valid_mels.pt")
torch.save(valid_labels, path_out + "valid_labels.pt")
torch.save(valid_classes, path_out +"valid_classes.pt")









