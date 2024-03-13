import os
import argparse
import subprocess

def reduce_volume(input_dir, output_dir, volume_reduction):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all WAV files in the input directory
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    for wav_file in wav_files:
        input_path = os.path.join(input_dir, wav_file)
        output_path = os.path.join(output_dir, wav_file)

        # Use ffmpeg to reduce volume
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-filter:a', f'volume={volume_reduction}',
            '-c:a', 'pcm_s16le',  # ensure output audio format is PCM 16-bit
            output_path
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    parser = argparse.ArgumentParser(description='Reduce volume of WAV files in a directory using ffmpeg')
    parser.add_argument('--input', help='Input directory containing WAV files')
    parser.add_argument('--output', help='Output directory to save processed WAV files')
    parser.add_argument('-r', '--reduction', type=float, default=0.5,
                        help='Volume reduction factor (default: 0.5, range: 0.0 to 1.0)')

    args = parser.parse_args()

    # Check if the input directory exists
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return

    reduce_volume(args.input, args.output, args.reduction)

if __name__ == "__main__":
    main()
