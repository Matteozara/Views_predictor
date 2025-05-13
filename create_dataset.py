# **
# It creates a dataset tarting from a folder of songs (inside "songs") and the respective views inside "BB_views.xlsx"
# It crates 4 mel spectrogram of 15 seconds for each song, and convert them to tensors together with labels.
# **

from pydub import AudioSegment
import os
import pandas as pd
import librosa
import numpy as np
import json
import torch
from sklearn.model_selection import train_test_split




def convert_sampling_rate(input_dir, output_dir, target_sr):
    for file_name in os.listdir(input_dir):
        print(file_name)
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(target_sr)
        audio.export(output_path, format="mp3")


def load_and_split_audio(audio_file, segment_length):
    y, sr = librosa.load(audio_file, sr=None)
    n_frames_15s = int(sr * segment_length) #calculate lenght mel

    audio_segments = []
    for i in range(0, len(y), n_frames_15s):
        segment = y[i:i + n_frames_15s]
        if len(segment) == n_frames_15s:
            audio_segments.append(segment)

    return audio_segments, sr

def encode_mel_spectrogram(input_path, out_path, segment_length=15, n_mels=128):
    audio_segments, sr = load_and_split_audio(input_path, segment_length) #split audio in 15 sec sub-audios
    mel_specs = []

    for segment in audio_segments:
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max) #convert to log scale


        '''# Resize to a fixed length if necessary
        # For example, to have a fixed length of 128 along the time axis
        fixed_length_mel_spec = np.zeros((n_mels, 128))
        if mel_spec_db.shape[1] >= 128:
            fixed_length_mel_spec = mel_spec_db[:, :128]
        else:
            fixed_length_mel_spec[:, :mel_spec_db.shape[1]] = mel_spec_db'''
        mel_specs.append(mel_spec_db)

    #save only 3 of them
    mel_to_save = []
    mel_to_save.append(mel_specs[3])
    mel_to_save.append(mel_specs[5])
    mel_to_save.append(mel_specs[7])


    if len(mel_specs) < 12:
        if len(mel_specs) < 9:
            mel_to_save[-1] = mel_specs[5]
            mel_to_save.append(mel_specs[len(mel_specs)-2])
            print("very very short, len: ", len(mel_specs))
        else:
            mel_to_save.append(mel_specs[len(mel_specs)-2])
            print("len mel spect list: ", len(mel_specs))
    else:
        mel_to_save.append(mel_specs[11])
    
    print("mel list audio 3: ", mel_to_save[0].shape)
    print("mel list audio 5: ", mel_to_save[1].shape)
    print("mel list audio 7: ", mel_to_save[2].shape)
    print("mel list audio 11: ", mel_to_save[3].shape)
    # Assume 'mel_spectrogram' is your mel spectrogram data
    np.save(out_path + '.npy', mel_to_save)


    return mel_specs


def is_not_number(a):
    return not isinstance(a, (int, float))









###SAME SAMPLE RATE
input_directory = "songs"
output_directory = "same_sr"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
target_sampling_rate = 44100  #44.1 kHz

convert_sampling_rate(input_directory, output_directory, target_sampling_rate)



###CREATE DATASET
in_dir = output_directory
out_dir = "dataset/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
json_path = "views.json"
views = {}
file_path = 'BB_views.xlsx'
df = pd.read_excel(file_path)

#print(df.head())

for i in range(1, 96):
    row_data = df.iloc[i]
    print(row_data["Song"])
    #print(row_data["View"])
    if is_not_number(row_data["View"]):
        continue
    if row_data["View"] != 0:
        mel_spectrograms = encode_mel_spectrogram(in_dir + "/" + row_data["Song"] + ".mp3", out_dir + row_data["Song"])
        #mel_spectrograms = encode_songs_into_mels(in_dir + row_data["Song"] + ".mp3", out_dir + row_data["Song"])#split_and_convert_to_mel(in_dir + row_data["Song"] + ".mp3", out_dir + row_data["Song"])
        views[row_data["Song"]] = int(row_data["View"])
        #print("Number of segments:", len(mel_spectrograms))
        print("################# finished a song ##################à")

#save json with views
with open(json_path, 'w') as json_file:
    json.dump(views, json_file)
print("final json saved!")




###CREATE LISTS
path_root = out_dir

path_out = "lists/"
if not os.path.exists(path_out):
    os.mkdir(path_out)

lista_songs = []
lista_class = []
lista_labels = []
flag = 0

with open(json_path, 'r') as json_file:
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









