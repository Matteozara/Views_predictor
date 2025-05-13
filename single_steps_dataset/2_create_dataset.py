import pandas as pd
import librosa
import numpy as np
import json


'''def split_and_convert_to_mel(audio_file, name, split_length=20, overlap=0.5, n_mels=128):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Calculate split length in samples
    split_length_samples = int(split_length * sr)

    # Calculate overlap length in samples
    overlap_samples = int(split_length_samples * overlap)

    # Split the audio into segments
    segments = []
    for start in range(0, len(y), split_length_samples - overlap_samples):
        segment = y[start:start + split_length_samples]
        # Ensure all segments have the same length
        if len(segment) < split_length_samples:
            segment = np.pad(segment, (0, split_length_samples - len(segment)))
        segments.append(segment)

    # Convert each segment to Mel spectrogram
    mel_spectrograms = []
    for segment in segments:
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
        mel_spectrograms.append(mel_spectrogram)

    print("mel spectrograms: ", len(mel_spectrograms))

    #save only 3 of them
    mel_to_save = []
    mel_to_save.append(mel_spectrograms[3])
    mel_to_save.append(mel_spectrograms[7])
    mel_to_save.append(mel_spectrograms[12])
    # Assume 'mel_spectrogram' is your mel spectrogram data
    np.save(name + '.npy', mel_to_save)

    print("saved mels: ", len(mel_to_save))

    #load it
    #mel_spectrogram = np.load('mel_spectrogram.npy')

    return mel_spectrograms'''


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







'''def preprocess_audio(audio_file, target_length, sr=22050):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=sr)
    
    # Ensure audio is at least target_length seconds long
    if len(y) < target_length * sr:
        # Pad audio with zeros if it's shorter than target length
        y = np.pad(y, (0, target_length * sr - len(y)), 'constant')
    elif len(y) > target_length * sr:
        # Trim audio if it's longer than target length
        y = y[:target_length * sr]
    
    return y

def extract_mel_spectrogram(audio, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    return mel_spec

def encode_songs_into_mels(audio_file, out_dir, target_length=240, sr=22050):
    # Preprocess audio to ensure it's target_length seconds long
    audio = preprocess_audio(audio_file, target_length, sr)
    
    # Split audio into 15-second segments
    segments = [audio[i:i+target_length*sr] for i in range(0, len(audio), target_length*sr)]
    
    # Extract Mel spectrogram from each segment
    all_mels = []
    for segment in segments:
        mel_spec = extract_mel_spectrogram(segment, sr)
        all_mels.append(mel_spec)
    

    # Convert list of Mel spectrograms into a single NumPy array
    all_mels = np.array(all_mels)
    print("shape: ", all_mels.shape)

    return all_mels'''

def is_not_number(a):
    return not isinstance(a, (int, float))


in_dir = "same_sr/"
out_dir = "dataset/"
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
        mel_spectrograms = encode_mel_spectrogram(in_dir + row_data["Song"] + ".mp3", out_dir + row_data["Song"])
        #mel_spectrograms = encode_songs_into_mels(in_dir + row_data["Song"] + ".mp3", out_dir + row_data["Song"])#split_and_convert_to_mel(in_dir + row_data["Song"] + ".mp3", out_dir + row_data["Song"])
        views[row_data["Song"]] = int(row_data["View"])
        #print("Number of segments:", len(mel_spectrograms))
        print("################# finished a song ##################à")


#save json with views
with open(json_path, 'w') as json_file:
    json.dump(views, json_file)
print("final json saved!")