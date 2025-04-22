import os
import subprocess
import sys
import argparse

def convert_mov_to_mp4(folder_path):
    """
    Converts all .mov files in the specified folder to .mp4 using ffmpeg.

    Args:
        folder_path (str): The path to the folder containing .mov files.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    print(f"Scanning folder: {folder_path}")

    # --- Get list of .mov files first to count them ---
    mov_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mov")]
    total_files = len(mov_files)

    if total_files == 0:
        print("No .mov files found in the specified folder.")
        return

    print(f"Found {total_files} .mov files to process.")
    # --- End change ---

    converted_count = 0
    skipped_count = 0
    error_count = 0
    current_file_index = 0 # --- Add counter ---

    # --- Iterate through the pre-filtered list ---
    for filename in mov_files:
        current_file_index += 1 # --- Increment counter ---
        input_path = os.path.join(folder_path, filename)
        output_filename = os.path.splitext(filename)[0] + ".mp4"
        output_path = os.path.join(folder_path, output_filename)

        # --- Print progress ---
        print(f"\n--- Processing file {current_file_index} of {total_files}: {filename} ---")
        # --- End change ---

        # Skip if output file already exists
        if os.path.exists(output_path):
            print(f"Skipping: Output file already exists: {output_path}")
            skipped_count += 1
            continue

        print(f"Converting: {input_path} -> {output_path}")
        try:
            # Basic ffmpeg command: ffmpeg -i input.mov output.mp4
            # -loglevel error: Suppresses verbose output except for errors
            # -nostdin: Prevents ffmpeg from reading from stdin
            command = [
                "ffmpeg",
                "-i", input_path,
                "-loglevel", "error", # Keep output clean
                "-nostdin",            # Avoid issues in batch processing
                # Add more ffmpeg options here if needed, e.g., codec settings
                # "-vcodec", "h264",
                # "-acodec", "aac",
                output_path
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Successfully converted: {output_filename}")
            converted_count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error converting {filename}: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            error_count += 1
            # Optionally remove partially created output file on error
            if os.path.exists(output_path):
                os.remove(output_path)
        except FileNotFoundError:
            print("Error: ffmpeg command not found. Make sure ffmpeg is installed and in your PATH.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during conversion of {filename}: {e}")
            error_count += 1

    print("\nConversion Summary:")
    print(f"  Total .mov files found: {total_files}") # --- Use total_files ---
    print(f"  Successfully converted: {converted_count}")
    print(f"  Skipped (already exist): {skipped_count}")
    print(f"  Errors: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .mov files to .mp4 in a specified folder using ffmpeg.")
    parser.add_argument("--folder", help="Path to the folder containing .mov files.", default="data/iphone_left")
    args = parser.parse_args()

    convert_mov_to_mp4(args.folder) 