"""script to take corpus and run pitch shifts on each file. 
Give a list of pitches (-p 3 6 12 -12 -5) 

todo - optionally concatenate the files together.
todo - optionally concatenate the shifted files together with the original files
todo - generate a frequency plot of the original corpus
todo - generate a frequency plot of the shifted corpus

NEEDS python3, and 
pip(3) install pyrubberband
pip3 install soundfile

run the script:
python3 -i path/to/corpus -p -5 11 12 -24 -o wav

thanks to chatGPT for some help"""

# import your libs
import os
import pyrubberband
import soundfile as sf
import argparse

# deal with arguments

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='provide a folder of sound files', default='none')
parser.add_argument ('-p', '--pitches', type=str, nargs='*', help="supply a list of pitches", default=[12,-12])
parser.add_argument('-o', '--output', type=str, help='select output type', default='wav')
args = parser.parse_args()


# Input directory
input_folder = args.input

# create output directory for pitch shifted files if it doesn't already exist
output_folder = os.path.join(input_folder, 'shiftedLib')
print(output_folder)
isExist = output_folder
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(output_folder)
   print("The new directory is created!")

# Ensure output folder exists, create if not
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process audio files with pitch shift and save with pitch value in filename
def process_audio_with_pitch_shift(input_file, output_folder, semitones):
    y, sr = sf.read(input_file)
    # Perform pitch shift using pyrubberband
    y_shifted = pyrubberband.pitch_shift(y, sr=sr, n_steps=semitones)

    # Generate output filename with pitch shift value
    output_filename = os.path.splitext(os.path.basename(input_file))[0] + f"_pitch_{semitones}.{args.output}"
    output_path = os.path.join(output_folder, output_filename)

    # Save processed audio to output file
    sf.write(output_path, y_shifted, sr, format=args.output)
    print(f"File {output_filename} has been written, noice")

# Iterate through files in the input folder
for filename in os.listdir(input_folder):
    input_file_path = os.path.join(input_folder, filename)

    # Check if the file is an audio file
    if filename.lower().endswith(('.mp3', '.wav', '.flac', '.aiff', '.aif')):
        # Perform multiple pitch shifts and save processed files
        for semitones in args.pitches:  # List of pitch shift values
            process_audio_with_pitch_shift(input_file_path, output_folder, semitones)

    else:
        print(f"Skipping {filename}: Not a supported audio format.")
    

print("Batch processing completed.")

# finally concatenate all of the files together and cleanup