import os, argparse
import librosa
import soundfile as sf


def clean(source_folder, destination_folder):
    print("Calculating dataset size...")

    total_size = 0
    processed_size = 0

    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(("wav", "opus", "mp3", "aac", "flac")):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
    
    print(f"Cleaning dataset...")

    idx = 0

    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(("wav", "opus", "mp3", "aac", "flac")) and not file.startswith("."):
                source_path = os.path.join(root, file)

                file = file.split(".")[0] + ".wav"
                file = str(idx).zfill(8) + "_" + file
                destination_path = os.path.join(destination_folder, file)

                try:
                    audio_data, sr = sf.read(source_path)
                except:
                    print("File " + source_path + " couldn't be read.")
                    continue

                # Convert to mono
                if audio_data.ndim == 2 and audio_data.shape[1] == 2:
                    audio_data= audio_data.mean(axis=1)

                # Resample to 44100
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=44100)

                # Converto to int16
                audio_data = (audio_data * 32767).astype("int16")
                
                # Save file to new location
                sf.write(destination_path, audio_data, samplerate=44100, format="wav", subtype="PCM_16")

                processed_size = processed_size + os.path.getsize(source_path)
                idx = idx + 1
                print("Processed " + str(int(processed_size / total_size * 100)) + "%", end="\r")

    print(f"\nDone.")


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
    
    