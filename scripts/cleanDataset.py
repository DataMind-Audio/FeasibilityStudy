import os, argparse, subprocess
import zipfile
import time


class Cleaner():
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination

        self.start_time = 0
        self.end_time = 0

        self.unprocessed_size = 0
        self.processed_size = 0

        self.ffmpeg_logs = 0
        self.formats = ("wav", "opus", "mp3", "aac", "flac")
        self.procs = []
        self.expand_zip()

    def expand_zip(self):
        for root, _, files in os.walk(self.source):
            for file in files:
                if file.endswith(".zip"):
                    with zipfile.ZipFile(os.path.join(root, file), "r") as zip_ref:
                        file = file[:-4]
                        zip_ref.extractall(os.path.join(root, file))

    def get_dataset_size(self, source_folder):
        size = 0

        for root, _, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                size += file_size
        
        return size

    def get_dataset_length(self, source_folder, sample_rate=44100):
        size = self.get_dataset_size(source_folder)
        length = size / (sample_rate * 2)
        return length

    def clean(self):
        print("Calculating dataset size...")

        self.start_time = time.time()

        self.unprocessed_size = self.get_dataset_size(self.source)

        print(f"Cleaning dataset...")

        idx = 0
        self.ffmpeg_logs = open("ffmpeg_log.txt", "w")

        for root, _, files in os.walk(self.source):
            for file in files:
                self.process_audio(root, file, idx)
        
        print(f"Waiting for {len(self.procs)} processes to finish...", end="")
        for p in self.procs:
            p.wait()

        os.remove("ffmpeg_log.txt")
        print(f"\nDone.")

        self.end_time = time.time()
    
    def process_audio(self, root, file, idx):
        source_path = os.path.join(root, file)
        self.processed_size = self.processed_size + os.path.getsize(source_path)

        if file.endswith(self.formats) and not file.startswith("."):
            file = file.split(".")[0] + ".wav"
            file = str(idx).zfill(8) + "_" + file
            destination_path = os.path.join(self.destination, file)

            # Remove silence
            call_ffmpeg = [
                "ffmpeg",
                '-i',
                source_path,
                '-af',
                f'silenceremove=stop_periods=-1:stop_duration={3}:stop_threshold={-60}dB',
                '-ac',
                '1', 
                '-c:a',
                'pcm_s16le', 
                '-y',
                destination_path,
            ]

            p = subprocess.Popen(call_ffmpeg, stderr=self.ffmpeg_logs)
            self.procs.append(p)

            max_procs = 32
            while len(self.procs) >= max_procs:
                for p in self.procs:
                    p.wait()
                    self.procs.remove(p)
                    break

            for p in self.procs:
                if p.poll() is not None:
                    self.procs.remove(p)

            idx = idx + 1
        
        print("Processed " + str(int(self.processed_size / self.unprocessed_size * 100)) + "%", end="\r")

    
    def log_results(self):
        print("Process took " + str(int(self.end_time - self.start_time)) + " seconds.")
        print("Processed dataset size: " + str(round(self.get_dataset_size(destination_folder) / 1000**3, 2)) + " GBs.")
        print("Processed dataset length " + str(round(self.get_dataset_length(destination_folder) / 60**2, 2)) + " hours." )


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

    cleaner = Cleaner(source_folder, destination_folder)
    cleaner.clean()
    cleaner.log_results()


