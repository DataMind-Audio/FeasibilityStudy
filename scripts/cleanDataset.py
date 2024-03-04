import os, argparse, subprocess
import zipfile
import time
import soundfile as sf


class Cleaner():
    def __init__(self, source, destination, expand=False):
        self.source = source
        self.destination = destination

        self.silence_threshhold = -70
        self.silence_duration = 3

        self.start_time = 0
        self.end_time = 0

        self.expand = expand

        self.unprocessed_size = 0
        self.processed_size = 0

        self.ffmpeg_logs = 0
        self.formats = ("wav", "opus", "mp3", "aac", "flac", "aiff", "aif")
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
    
    def get_ffmpeg_call(self, source_path, destination_path, config):
        # Sum to mono
        if config == 0:
            destination_path = destination_path[:-4] + "_sumToMono" + destination_path[-4:]
            call = [
                "ffmpeg",
                "-i",
                source_path,
                "-af",
                f'pan=mono|c0=0.5*c0+0.5*c1,silenceremove=stop_periods=-1:stop_duration={self.silence_duration}:stop_threshold={self.silence_threshhold}dB',
                '-c:a',
                'pcm_s16le',
                "-y",
                destination_path
            ]

        # Sum to mono and invert phase
#        elif config == 1:
#            destination_path = destination_path[:-4] + "_sumToMonoPhaseInv" + destination_path[-4:]
#            call = [
#                "ffmpeg",
#                "-i",
#                source_path,
#                "-af",
#                f'pan=mono|c0=c0+c1,aeval=-val(0),silenceremove=stop_periods=-1:stop_duration={self.silence_duration}:stop_threshold={self.silence_threshhold}dB',
#                '-c:a',
#                'pcm_s16le',
#                "-y",
#                destination_path
#            ]

        # Take left channel
        elif config == 1:
            destination_path = destination_path[:-4] + "_left" + destination_path[-4:]
            call = [
                "ffmpeg",
                "-i",
                source_path,
                "-af",
                f'pan=mono|c0=c0,silenceremove=stop_periods=-1:stop_duration={self.silence_duration}:stop_threshold={self.silence_threshhold}dB',
                '-c:a',
                'pcm_s16le',
                "-y",
                destination_path
            ]
        
        # Take left channel and invert phase
#        elif config == 3:
#            destination_path = destination_path[:-4] + "_leftInv" + destination_path[-4:]
#            call = [
#                "ffmpeg",
#                "-i",
#                source_path,
#                "-af",
#                f'pan=mono|c0=c0,aeval=-val(0),silenceremove=stop_periods=-1:stop_duration={self.silence_duration}:stop_threshold={self.silence_threshhold}dB',
#                '-c:a',
#                'pcm_s16le',
#                "-y",
#               destination_path
#            ]
        
        # Take right channel
        elif config == 2:
            destination_path = destination_path[:-4] + "_right" + destination_path[-4:]
            call = [
                "ffmpeg",
                "-i",
                source_path,
                "-af",
                f'pan=mono|c0=c1,silenceremove=stop_periods=-1:stop_duration={self.silence_duration}:stop_threshold={self.silence_threshhold}dB',
                '-c:a',
                'pcm_s16le',
                "-y",
                destination_path
            ]
        
        # Take right channel and invert phase
#        elif config == 5:
#            destination_path = destination_path[:-4] + "_rightInv" + destination_path[-4:]
#            call = [
#                "ffmpeg",
#                "-i",
#                source_path,
#                "-af",
#                f'pan=mono|c0=c1,aeval=-val(0),silenceremove=stop_periods=-1:stop_duration={self.silence_duration}:stop_threshold={self.silence_threshhold}dB',
#                '-c:a',
#                'pcm_s16le',
#                "-y",
#                destination_path
#            ]
        
        return call
        

    def clean(self):
        print("Calculating dataset size...")

        self.start_time = time.time()

        self.unprocessed_size = self.get_dataset_size(self.source)

        print(f"Cleaning dataset...")

        idx = 0
        self.ffmpeg_logs = open("ffmpeg_log.txt", "w") 

        for root, _, files in os.walk(self.source):
            for file in files:
                idx = self.process_audio(root, file, idx)
        
        print(f"Waiting for {len(self.procs)} processes to finish...", end="")
        for p in self.procs:
            p.wait()
        self.ffmpeg_logs.close()
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

            try:
                audio, sr = sf.read(source_path, always_2d=True)
            except:
                print(f'Failed to process file: {source_path}')
                return idx

            end = 3

            if audio[1].size == 1:
                end =  2
            
            if self.expand == False:
                end = 1 
            
            for config in range(0, end):
                p = subprocess.Popen(self.get_ffmpeg_call(source_path, destination_path, config), stderr=self.ffmpeg_logs)
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

        return idx

    
    def log_results(self):
        print("Process took " + str(int(self.end_time - self.start_time)) + " seconds.")
        print("Processed dataset size: " + str(round(self.get_dataset_size(destination_folder) / 1000**3, 2)) + " GBs.")
        print("Processed dataset length " + str(round(self.get_dataset_length(destination_folder) / 60**2, 2)) + " hours." )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input dataset folder', required=True)
    parser.add_argument('-o', '--output', type=str, help='output dataset folder', required=True)
    parser.add_argument('-e', '--expand', type=str, help="extracts channels and flips phase", default="n")
    args = parser.parse_args()

    source_folder = args.input
    destination_folder = args.output
    
    expand = False

    if args.expand == "y":
        expand = True
    
    if not os.path.exists(destination_folder):
      os.makedirs(destination_folder)

    cleaner = Cleaner(source_folder, destination_folder, expand)
    cleaner.clean()
    cleaner.log_results()


