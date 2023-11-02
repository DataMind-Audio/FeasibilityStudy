import subprocess, os, argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='provide a folder filled with sound files', default='None')
parser.add_argument('-o', '--output', type=str, help='provide a folder path for the outputs', default='None')
parser.add_argument ('-t', '--threshold', type=float, help="set the silence detection threhold", default=-50.)
parser.add_argument('-s', '--silenceduration', type=float, help='set the min silence duration to remove, in seconds', default='3')
parser.add_argument('-f', '--ffmpegpath', type=str, help='specify the path to ffmpeg', default=None)
args = parser.parse_args()

# deal with ffmpeg path, this is important for windoze users who often struggle to persuade windows to know where ffmpeg is
if args.ffmpegpath == None:
    ffmpeg_path='ffmpeg' 
else:
    ffmpeg_path=arg.ffmpegpath 

# now sort the other arguments 

thresh=args.threshold
duration=args.silenceduration
input_folder=args.input
extension='wav'
# make an output folder if it doesn't exist 
if args.output == None:
    output_folder = os.path.join(input_folder, 'silenceStrippedResults')
    print(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("The new directory is created!", output_folder)
else:
    output_folder = args.output


# here's a definition we can use again and again to remove silences with FFMPEG. 

def removeSilences(inputPath, outputPath, minimum_silence_duration, silence_threshold, ffmpeg_path):
    
    call_ffmpeg = [
        ffmpeg_path,
        '-i',
        inputPath,
        '-af',
        f'silenceremove=stop_periods=-1:stop_duration={minimum_silence_duration}:stop_threshold={silence_threshold}dB',
        '-ac',
        '1',
        outputPath,
        '-y'
    ]    

    subprocess.call(call_ffmpeg)

# here we loop through directories supplied with input_folder, recreate the folder structure and run the process on each audio file that it finds

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
            removeSilences(input_file_path, output_file_name, duration, thresh, ffmpeg_path)