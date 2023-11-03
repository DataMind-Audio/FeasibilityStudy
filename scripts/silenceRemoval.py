import subprocess, os, argparse


if __name__ == "__main__":
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
        ffmpeg_path=args.ffmpegpath 

    # now sort the other arguments 

    thresh=args.threshold
    duration=args.silenceduration
    input_folder=args.input
    extension='wav'
    # make an output folder if it doesn't exist 
    if args.output == 'None':
        output_folder = input_folder + "_silenceRemoved"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print("The new directory is created!", output_folder)
    else:
        output_folder = args.output

    total_size = 0
    unprocessed_size = 0
    processed_size = 0

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(("wav")):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size

    # Remove silence
    print("Removing silence.")

    for root, _, files in os.walk(input_folder):
        for file in files:
            source_path = os.path.join(root, file)

            destination_path = os.path.join(output_folder, file)

            with open("ffmpeg_log.txt", "w") as log_file:
        
                call_ffmpeg = [
                    ffmpeg_path,
                    '-i',
                    source_path,
                    '-af',
                    f'silenceremove=stop_periods=-1:stop_duration={duration}:stop_threshold={thresh}dB',
                    '-ac',
                    '1',
                    destination_path,
                    '-y'
                ] 

                subprocess.call(call_ffmpeg, stderr=log_file)

                unprocessed_size = unprocessed_size + os.path.getsize(source_path)
                processed_size = processed_size + os.path.getsize(destination_path)
            
                print(
                    "Processed " + str(int(unprocessed_size / total_size * 100)) + "%" + 
                    "   Removed " + str(int(100 - processed_size / unprocessed_size  * 100)) + "% of dataset.", end="\r"
                )


    print("\n")
    os.remove("ffmpeg_log.txt")
