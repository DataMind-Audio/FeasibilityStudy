import os, argparse, subprocess
import librosa
import soundfile as sf
import time


def get_dataset_size(source_folder):
    size = 0

    for root, _, files in os.walk(source_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            size += file_size
    
    return size


def get_dataset_length(source_folder, sample_rate=44100):
    size = get_dataset_size(source_folder)
    length = size / (sample_rate * 2)
    return length


def clean(source_folder, destination_folder):
    print("Calculating dataset size...")

    start = time.time()

    total_size = get_dataset_size(source_folder)
    processed_size = 0
    silence_removed = 0

    print(f"Cleaning dataset...")

    idx = 0
    ffmpeg_logs = open("ffmpeg_log.txt", "w")

    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(("wav", "opus", "mp3", "aac", "flac")) and not file.startswith("."):
                source_path = os.path.join(root, file)

                file = file.split(".")[0] + ".wav"
                file = str(idx).zfill(8) + "_" + file
                destination_path = os.path.join(destination_folder, file)

                # Try open file - skip if corrupt
                try:
                    audio_data, sr = sf.read(source_path)
                except:
                    print("Error cleaning file. Skipping.")
                    continue

                # Convert to mono
                if audio_data.ndim == 2 and audio_data.shape[1] == 2:
                    audio_data= audio_data.mean(axis=1)

                # Resample to 44100
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=44100)

                # Converto to int16
                audio_data = (audio_data * 32767).astype("int16")
                
                # Save file to new location
                temp_file_name = destination_path + ".tmp"
                sf.write(temp_file_name, audio_data, samplerate=44100, format="wav", subtype="PCM_16")
                processed_size = processed_size + os.path.getsize(source_path)

                # Remove silence
                call_ffmpeg = [
                    "ffmpeg",
                    '-i',
                    temp_file_name,
                    '-af',
                    f'silenceremove=stop_periods=-1:stop_duration={3}:stop_threshold={-60}dB',
                    '-ac',
                    '1',
                    destination_path,
                    '-y'
                ] 

                subprocess.call(call_ffmpeg, stderr=ffmpeg_logs)

                silence_removed = silence_removed + os.path.getsize(temp_file_name) - os.path.getsize(destination_path) 

                os.remove(temp_file_name)

                print(
                    "Processed " + str(int(processed_size / total_size * 100)) + "%" + 
                    "          " + str(int(silence_removed / total_size  * 100)) + "% of dataset was silence.", end="\r"
                )

                idx = idx + 1

    os.remove("ffmpeg_log.txt")
    print(f"\nDone.")

    end = time.time()
    print("Process took " + str(int(end - start)) + " seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input dataset folder', default='None')
    parser.add_argument('-o', '--output', type=str, help='output dataset folder', default='None')
    args = parser.parse_args()

    source_folder = ""
    destination_folder = ""

    if args.input != "None":
        source_folder = args.input
    else:
        raise Exception("No input provided.")

    if args.output != "None":
        destination_folder = args.output
    else:
        destination_folder = source_folder + "_processed"
    
    if not os.path.exists(destination_folder):
      os.makedirs(destination_folder)

    clean(source_folder, destination_folder)

    print("Processed dataset size: " + str(round(get_dataset_size(destination_folder) / 1024**3, 2)) + " GBs.")
    print("Processed dataset length " + str(round(get_dataset_length(destination_folder) / 60**2, 2)) + " hours." )
