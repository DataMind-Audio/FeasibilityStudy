import os
import librosa
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='provide a folder of sound files', default='none')
args = parser.parse_args()

# Directory containing the sound files
corpus_folder = args.input

# Interval for averaging pitch values (in seconds)
interval = .1  # Change this to the desired interval

# List to store averaged MIDI pitch values
averaged_midi_pitch_values = []

# Function to extract MIDI pitch from audio file
def extract_midi_pitch(audio_file):
    y, sr = librosa.load(audio_file)
    hop_length = int(interval * sr)  # Number of samples in the interval
    for i in range(0, len(y), hop_length):
        segment = y[i:i+hop_length]
        pitches, magnitudes = librosa.core.piptrack(y=segment, sr=sr, n_fft=128)
        pitch_index = magnitudes.argmax(axis=0)
        pitch_values = pitches[pitch_index]
        if len(pitch_values) >= 0:
            averaged_midi_pitch_values.append(np.mean(librosa.hz_to_midi(pitch_values)))

# Iterate through files in the corpus folder
for filename in os.listdir(corpus_folder):
    audio_file_path = os.path.join(corpus_folder, filename)

    # Check if the file is an audio file
    if filename.lower().endswith(('.mp3', '.wav', '.flac', '.aiff', '.aif')):
        # Extract averaged MIDI pitch from the audio file
        extract_midi_pitch(audio_file_path)

    else:
        print(f"Skipping {filename}: Not a supported audio format.")

# Create a histogram of averaged MIDI pitch values
plt.figure(figsize=(60, 6))
plt.hist(averaged_midi_pitch_values, bins=np.arange(128)-0.5, alpha=0.7, edgecolor='black')
plt.xlabel('MIDI Pitch')
plt.ylabel('Frequency')
plt.title(f'Averaged MIDI Pitch Distribution Across the Corpus (Averaged over {interval} seconds)')
plt.xticks(np.arange(128))
plt.grid(True)
plt.show()