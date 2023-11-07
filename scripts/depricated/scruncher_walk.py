import librosa
import numpy as np
import soundfile as sf
import argparse, os

# todo deal with very short files 
# concatenate all together 
# enforce mono or create some useful mono from the two stereo inputs.

# deal with input arguments to the script

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='provide a folder filled with sound files', default='None')
parser.add_argument('-o', '--output', type=str, help='provide a folder path for the outputs', default='None')
parser.add_argument ('-c', '--chunksize', type=float, help="set the size of each chunk of audio you want to grab in seconds", default=1.0)
parser.add_argument('-p', '--percent', type=float, help='set the percentages of the corpus you want to grab', default='10')
parser.add_argument('-f', '--fadelen', type=int, help='set the length of the fade in and out of each chunk of audio in samples', default='25')
args = parser.parse_args()

# sort out arguments 

# percentage
percentage_of_corpus=args.percent/100
print("corpus percentage is", args.percent, percentage_of_corpus)
# input folder
input_folder=args.input

# chunkSize 
chunk_size = args.chunksize * 44100
print(chunk_size)

# fadeLength
fade_length = args.fadelen

# filetype for export
extension='wav'
bitdepth='PCM_16'



# create output directory for pitch shifted files if it doesn't already exist

if args.output == "None":
    output_folder = os.path.join(input_folder, 'scrunchedLibrary')
    print(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("The new directory is created!", output_folder)
else:
    output_folder = args.output

# Function to process each sound file

def process_sound_file(file_path, chunk_size, fade_length, percentage):
    # Load the sound file
    y, sr = librosa.load(file_path, sr=44100)
    
    # Calculate the total duration of the audio file
    total_duration = librosa.get_duration(y=y, sr=sr)
    print("total duration =", total_duration)
    # get the total length minus the chunk size, this way, you won't overshoot the end of the file
    total_duration_minus_chunk_len = total_duration-(chunk_size/sr)
    # Calculate 10% of the total duration
    ten_percent_duration = total_duration * percentage
    print("percentage of that duration =", ten_percent_duration)
    # Calculate the number of chunks needed to cover 10% of the file
    total_chunks = int(np.ceil(ten_percent_duration / (chunk_size / sr)))
    print("number of chunks you're going to get", total_chunks)

    # Initialize an empty array to store the chunks
    chunks = []
    
    # Iterate through the sound file and extract chunks
    for i in range(total_chunks):
        # Calculate start and end times for the current chunk
        if i == 0:
            start_time=0
            end_time=chunk_size/sr
        else:
            #print("eachChunk gap will be this long", (total_duration_minus_chunk_len/total_chunks))
            start_time = i * (total_duration_minus_chunk_len/total_chunks)
            end_time = start_time + (chunk_size/sr) 
        #print(start_time, end_time)
        # Extract the current chunk
        chunk_start_idx = int(start_time * sr)
        chunk_end_idx = int(end_time * sr)
        chunk = y[chunk_start_idx:chunk_end_idx]
        
        # Apply fade in and fade out
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        chunk[:fade_length] *= fade_in
        chunk[-fade_length:] *= fade_out
        
         # Add the chunk to the list
        chunks.append(chunk)
    
    # Concatenate all the chunks
    output_chunk = np.concatenate(chunks)
    
    return output_chunk, sr

# find all audio files in the folder and run the function on each file 
# addition in this script is that you walk through the entire directory of files and reproduce the directory structure in new folder. 
# important to specify the output folder here.
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(('.mp3', '.wav', '.flac', '.aiff', '.aif', '.WAV', '.AIF', '.AIFF')):
            input_file_path = os.path.join(root, file)
            print(input_file_path)
            relative_path = os.path.relpath(input_file_path, input_folder)
            output_folder_path = os.path.join(output_folder, os.path.dirname(relative_path))
            #output_folder_path = os.path.join(output_folder, os.path.abspath(root, input_file_path))
            print(output_folder_path)
            os.makedirs(output_folder_path, exist_ok=True)
            output_file_name = os.path.join(output_folder_path, f'{os.path.splitext(file)[0]}_scrunched.{extension}')
            print(output_file_name)
            # if is an audio file, run the process
            output_chunk, sr = process_sound_file(input_file_path, chunk_size, fade_length, percentage_of_corpus)
            sf.write(output_file_name, output_chunk, samplerate=44100, format='wav', subtype=bitdepth)

