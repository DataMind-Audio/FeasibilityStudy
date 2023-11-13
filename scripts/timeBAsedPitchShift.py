import pydub
from pydub import AudioSegment
import os
import argparse

# deal with arguments

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='provide a folder of sound files', default='none')
parser.add_argument ('-p', '--pitches', type=str, nargs='*', help="supply a list of pitches", default=[12,-12])
parser.add_argument('-o', '--output', type=str, help='select output type', default='wav')
args = parser.parse_args()


def pitch_shift_wav(input_folder, output_folder, semitones):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all WAV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            # Load the audio file
            input_path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_file(input_path, format="wav")

            # Perform time-based pitch shifting
            shifted_audio = audio.speedup(playback_speed=(2 ** (semitones / 12)))

            # Create the output path
            output_path = os.path.join(output_folder, filename)

            # Export the shifted audio to a new WAV file
            shifted_audio.export(output_path, format="wav")

            print(f"Pitch shifting {filename} completed.")

if __name__ == "__main__":
    # Set your input and output folders
    input_folder = args.input
    output_folder = args.output

    # Set the pitch shift in semitones
    semitones = args.pitches  # Adjust this value based on your preference

    # Perform pitch shifting
    pitch_shift_wav(input_folder, output_folder, semitones)
