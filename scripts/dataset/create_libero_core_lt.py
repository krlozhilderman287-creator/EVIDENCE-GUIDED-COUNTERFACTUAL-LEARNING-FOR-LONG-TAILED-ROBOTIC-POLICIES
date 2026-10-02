import h5py
import random
import argparse
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

# --- Configure logging for clear output ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_demo_info(filepath: Path, group_path: str, task_order: list[str]) -> dict | None:
    """
    Opens an HDF5 file once to get the number of demos, average length, and its order.
    
    Returns:
        A dictionary with file info or None if the file is invalid.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            demo_group = f.get(group_path)
            if not demo_group:
                logging.warning(f"Group '{group_path}' not found in {filepath.name}")
                return None
            
            demo_keys = list(demo_group.keys())
            demo_count = len(demo_keys)
            if demo_count == 0:
                return None

            # Calculate average length
            total_length = sum(demo_group[name]['actions'].shape[0] for name in demo_keys)
            avg_length = total_length / demo_count

            # Determine task order
            task_name = filepath.name.replace('_demo.hdf5', '')
            order = task_order.index(task_name) if task_name in task_order else -1

            return {'path': filepath, 'count': demo_count, 'length': avg_length, 'order': order}

    except Exception as e:
        logging.error(f"Could not read file {filepath.name}: {e}")
        return None


def truncated_pareto_inverse_cdf(u: np.ndarray, alpha: float, min_v: float, max_v: float) -> np.ndarray:
    """
    Calculates values from the inverse CDF of a truncated Pareto distribution.
    This helps create a long-tailed distribution of data points.
    """
    term1 = 1 - (min_v / max_v)**alpha
    term2 = 1 - u * term1
    return min_v / (term2**(1 / alpha))


def process_single_hdf5_file(source_path: Path, dest_path: Path, group_path: str, num_to_exclude: int) -> bool:
    """
    Copies an HDF5 file, excluding a random subset of demos.
    """
    try:
        with h5py.File(source_path, 'r') as source_f, h5py.File(dest_path, 'w') as dest_f:
            source_container = source_f[group_path]
            dest_container = dest_f.create_group(group_path)

            all_demo_names = list(source_container.keys())
            
            # Decide which demos to exclude
            demos_to_exclude = set(random.sample(all_demo_names, k=num_to_exclude)) if num_to_exclude > 0 else set()
            
            # Copy the remaining demos
            demos_to_keep = [name for name in all_demo_names if name not in demos_to_exclude]
            for demo_name in demos_to_keep:
                source_container.copy(source_container[demo_name], dest_container, name=demo_name)
        
        logging.debug(f"Kept {len(demos_to_keep)} demos in {dest_path.name}")
        return True
    except Exception as e:
        logging.error(f"Failed to process {source_path.name}: {e}")
        return False


def create_long_tail_dataset(source_dir: Path, dest_dir: Path, task_order: list[str], group_path: str, 
                             alpha: float, min_demos: int, max_demos: int, seed: int):
    """
    Main execution function to create a long-tailed version of an HDF5 dataset.
    """
    logging.info("--- Starting HDF5 dataset processing for long-tail distribution ---")
    random.seed(seed)

    # --- Phase 1: Scan all files and get their metadata ---
    logging.info("[1/3] Scanning all source files...")
    source_files = list(source_dir.glob("*.hdf5"))
    if not source_files:
        logging.error(f"No .hdf5 files found in {source_dir}. Exiting.")
        return

    file_info_list = [info for f_path in tqdm(source_files, desc="Scanning files") if (info := get_demo_info(f_path, group_path, task_order))]
    
    # Sort files based on the predefined task order
    file_info_list.sort(key=lambda x: x['order'])
    logging.info(f"Successfully scanned and sorted {len(file_info_list)} valid HDF5 files.")

    # --- Phase 2: Calculate how many demos to keep for each file ---
    logging.info("[2/3] Calculating the number of demos to keep per file...")
    num_files = len(file_info_list)
    quantiles = np.linspace(0, 1, num_files)
    
    # Generate Pareto-distributed values and sort them in descending order
    pareto_values = truncated_pareto_inverse_cdf(quantiles, alpha, min_demos, max_demos)
    num_to_keep_array = np.round(pareto_values).astype(int)
    num_to_keep_array = np.sort(num_to_keep_array)[::-1] # Sort descending
    
    logging.info(f"Target demos per file (descending order): {num_to_keep_array}")

    # --- Phase 3: Process each file and generate the new dataset ---
    logging.info(f"[3/3] Processing files and saving to {dest_dir}...")
    dest_dir.mkdir(exist_ok=True)
    
    success_count = 0
    for idx, file_info in enumerate(tqdm(file_info_list, desc="Processing files")):
        total_demos = file_info['count']
        
        # Ensure we don't try to keep more demos than available
        num_to_keep = min(total_demos, num_to_keep_array[idx])
        num_to_exclude = total_demos - num_to_keep
        
        source_path = file_info['path']
        dest_path = dest_dir / source_path.name

        if process_single_hdf5_file(source_path, dest_path, group_path, num_to_exclude):
            success_count += 1
            
    logging.info(f"--- Processing complete! ---")
    logging.info(f"Successfully created {success_count} / {len(file_info_list)} new HDF5 files in {dest_dir}")


def main():
    """Parses command-line arguments and runs the dataset creation process."""
    parser = argparse.ArgumentParser(
        description="Create a long-tailed (LT) version of a LIBERO HDF5 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Define Command-Line Arguments ---
    parser.add_argument('--source_dir', type=Path, required=True, help="Source directory containing the original HDF5 files.")
    parser.add_argument('--group_path', type=str, default='data', help="The key/group inside the HDF5 file where demos are stored.")
    parser.add_argument('--alpha', type=float, default=0.7, help="Shape parameter 'alpha' for the Pareto distribution. Controls the steepness of the tail.")
    parser.add_argument('--min_demos', type=int, default=5, help="Minimum number of demos a file can have after processing.")
    parser.add_argument('--max_demos', type=int, default=46, help="Maximum number of demos a file can have after processing.")
    parser.add_argument('--seed', type=int, default=100, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # --- Define the canonical order of tasks ---
    # This is crucial for applying the long-tail distribution correctly.
    task_order = [
        'pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate',
        'pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate',
        'pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate',
        'pick_up_the_ketchup_and_place_it_in_the_basket',
        'pick_up_the_alphabet_soup_and_place_it_in_the_basket',
        'push_the_plate_to_the_front_of_the_stove',
        'put_the_bowl_on_top_of_the_cabinet',
        'put_the_cream_cheese_in_the_bowl',
        'put_the_wine_bottle_on_top_of_the_cabinet',
        'put_the_wine_bottle_on_the_rack'
    ]

    # --- Run the main logic ---
    dest_dir = Path(str(args.source_dir).replace('full', 'lt'))
    create_long_tail_dataset(
        source_dir=args.source_dir,
        dest_dir=dest_dir,
        task_order=task_order,
        group_path=args.group_path,
        alpha=args.alpha,
        min_demos=args.min_demos,
        max_demos=args.max_demos,
        seed=args.seed
    )

if __name__ == "__main__":
    main()