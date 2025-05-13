#file that converts all the songs to the same sample rate
from pydub import AudioSegment
import os

def convert_sampling_rate(input_dir, output_dir, target_sr):
    for file_name in os.listdir(input_dir):
        print(file_name)
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(target_sr)
        audio.export(output_path, format="mp3")

# Example usage
input_directory = "songs"
output_directory = "same_sr"
target_sampling_rate = 44100  #44.1 kHz

convert_sampling_rate(input_directory, output_directory, target_sampling_rate)
