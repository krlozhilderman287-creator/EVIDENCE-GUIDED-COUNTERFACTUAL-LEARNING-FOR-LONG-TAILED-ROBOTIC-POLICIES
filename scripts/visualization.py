import h5py
import mediapy
import numpy as np
import argparse
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image, ImageDraw, ImageFont

def write_video(
    images: Union[np.ndarray, List[np.ndarray]],
    path: Union[str, Path],
    texts: Optional[List[str]] = None,
    fps: int = 10
):
    """
    Saves a sequence of images as a GIF, with optional text overlay.
    """
    # If text is provided, draw it on each image.
    if texts is not None:
        if len(images) != len(texts):
            raise ValueError("The number of images and texts must match.")
        
        # This list comprehension creates a new list of images with text.
        font = ImageFont.load_default()
        processed_images = []
        for i, img_array in enumerate(images):
            img = Image.fromarray(img_array)
            draw = ImageDraw.Draw(img)
            # Draw text with a small margin (10, 10) from the top-left corner.
            draw.text((10, 10), texts[i], (255, 255, 255), font=font)
            processed_images.append(np.array(img))
        images = processed_images

    # Write the final image sequence to a GIF file.
    mediapy.write_video(path, images, fps=fps, codec='gif')


def convert_hdf5_to_gifs(source_dir: Union[str, Path], fps: int):
    """
    Converts all demonstrations within HDF5 files from a source directory into GIFs.
    """
    source_path = Path(source_dir)
    # Check if the source directory exists.
    if not source_path.is_dir():
        print(f"Error: Source directory not found at '{source_path}'")
        return

    gifs_path = source_path / "gifs"
    gifs_path.mkdir(exist_ok=True)  # Create the output directory if it doesn't exist.

    # Use .glob() to find all HDF5 files.
    hdf5_files = list(source_path.glob("*.hdf5"))
    if not hdf5_files:
        print(f"Warning: No .hdf5 files found in {source_path}")
        return

    print(f"Found {len(hdf5_files)} HDF5 files. Starting conversion to GIFs...")

    # Process each HDF5 file found.
    for file_path in hdf5_files:
        try:
            with h5py.File(file_path, "r") as f:
                # Check if 'data' group exists
                if 'data' not in f:
                    print(f"Warning: 'data' group not found in {file_path}. Skipping.")
                    continue
                
                # Iterate over each demonstration (e.g., 'demo_0', 'demo_1') in the file.
                for demo_key in f["data"].keys():
                    # Load the image sequence. The '[:]' loads the data into a NumPy array.
                    image_sequence = f[f"data/{demo_key}/obs/agentview_rgb"][:]
                    
                    # Vectorized operation: Flip all images vertically at once.
                    # This is much faster than looping through each image.
                    images = np.flip(image_sequence, axis=1)

                    gif_name = f"{file_path.stem[:-5]}_{demo_key}.gif"
                    gif_path = gifs_path / gif_name
                    
                    # Save the processed images as a GIF.
                    write_video(images, gif_path, fps=fps)
                    print(f"Successfully saved GIF: {gif_path}")

        except Exception as e:
            # Catch potential errors during file processing and report them.
            print(f"Error processing file {file_path}: {e}")


def main():
    # Set up the argument parser to handle command-line inputs.
    parser = argparse.ArgumentParser(
        description="Convert demonstrations in HDF5 files from a specified directory to GIFs."
    )
    
    # Add a required positional argument for the source directory.
    parser.add_argument(
        "--source-dir",
        default= 'dataset_all/libero_core_lt_no_noops_target_apa',
        type=str,
        help="The path to the directory containing the .hdf5 dataset files."
    )
    
    # Add an optional argument for FPS with a default value.
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the output GIFs (default: 10)."
    )
    
    # Parse the arguments from the command line.
    args = parser.parse_args()
    
    # Call the main processing function with the provided arguments.
    convert_hdf5_to_gifs(args.source_dir, args.fps)


if __name__ == "__main__":
    # This block ensures the script runs only when executed directly from the terminal.
    main()