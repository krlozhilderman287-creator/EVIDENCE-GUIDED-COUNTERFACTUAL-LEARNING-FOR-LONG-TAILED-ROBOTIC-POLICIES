import shutil
import textwrap
from pathlib import Path
import logging
import argparse

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def copy_dataset_files(dataset_root: Path, target_dir_name: str, task_suite_names: list[str], task_descriptions: list[str]):
    """
    Core logic to find and copy specified HDF5 task dataset files.
    
    Args:
        dataset_root (Path): The root directory of the source datasets.
        target_dir_name (str): The name of the target directory to create inside the root.
        task_suite_names (list[str]): A list of source task suite folder names.
        task_descriptions (list[str]): A list of task descriptions to identify the files to be copied.
    """
    target_dataset_dir = dataset_root / target_dir_name

    # 1. (Efficient) First, scan all source directories to build a map 
    #    from filenames to their full paths.
    logging.info("Scanning source directories to build a file map...")
    source_file_map = {}
    for suite_name in task_suite_names:
        folder_path = dataset_root / f'{suite_name}_no_noops'
        if not folder_path.is_dir():
            logging.warning(f"Directory not found, skipping: {folder_path}")
            continue
        
        for hdf5_path in folder_path.glob('*.hdf5'):
            source_file_map[hdf5_path.name] = hdf5_path
    
    logging.info(f"Found {len(source_file_map)} total source HDF5 files.")

    # 2. Ensure the target directory exists.
    target_dataset_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Target directory is: {target_dataset_dir}")

    # 3. Iterate through the target task list and perform copy operations.
    copied_count = 0
    skipped_count = 0
    not_found_count = 0

    for task_desc in task_descriptions:
        hdf5_name = f'{task_desc}_demo.hdf5'
        target_hdf5_path = target_dataset_dir / hdf5_name

        source_path = source_file_map.get(hdf5_name)

        if not source_path:
            logging.warning(f"Source file not found for task: {task_desc}")
            not_found_count += 1
            continue

        if not target_hdf5_path.exists():
            # (Robust) Use shutil.copy2 for file copying.
            try:
                shutil.copy2(source_path, target_hdf5_path)
                logging.info(f"Copied: {source_path.name} -> {target_hdf5_path}")
                copied_count += 1
            except Exception as e:
                logging.error(f"Failed to copy {source_path.name}. Error: {e}")
        else:
            logging.info(f"Exists, skipping copy for: {target_hdf5_path.name}")
            skipped_count += 1

    logging.info("=" * 20 + " Summary " + "=" * 20)
    logging.info(f"Tasks processed: {len(task_descriptions)}")
    logging.info(f"Files copied: {copied_count}")
    logging.info(f"Files skipped (already exist): {skipped_count}")
    logging.info(f"Files not found in source: {not_found_count}")


def main():
    """
    Parses command-line arguments and runs the dataset copy process.
    """
    parser = argparse.ArgumentParser(
        description="Find and copy specific LIBERO dataset files based on task descriptions.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )

    parser.add_argument(
        '--dataset_root',
        type=Path,
        default=Path("dataset_all/"),
        help="The root directory where source datasets are located and the target directory will be created.\n(default: dataset_all/)"
    )
    
    parser.add_argument(
        '--target_dir_name',
        type=str,
        default='libero_core_full_no_noops',
        help="Name of the directory inside '--dataset_root' where files will be copied.\n(default: LIBERO_Core_FULL)"
    )

    parser.add_argument(
        '--task_suites',
        type=str,
        nargs='+', # This allows accepting one or more arguments
        default=['libero_spatial', 'libero_goal', 'libero_object'],
        help="A list of source task suite names (without the '_no_noops' suffix).\n(default: libero_spatial libero_goal libero_object)"
    )

    args = parser.parse_args()
    
    # Use textwrap.dedent to handle multiline strings without leading indentation from the source code.
    target_task_descriptions = textwrap.dedent('''
        pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
        pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
        pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
        pick_up_the_ketchup_and_place_it_in_the_basket
        pick_up_the_alphabet_soup_and_place_it_in_the_basket
        push_the_plate_to_the_front_of_the_stove
        put_the_bowl_on_top_of_the_cabinet
        put_the_cream_cheese_in_the_bowl
        put_the_wine_bottle_on_top_of_the_cabinet
        put_the_wine_bottle_on_the_rack
    ''').strip().splitlines()
    
    # Call the main logic function with the parsed arguments
    copy_dataset_files(
        dataset_root=args.dataset_root,
        target_dir_name=args.target_dir_name,
        task_suite_names=args.task_suites,
        task_descriptions=target_task_descriptions
    )


# Standard practice: protect the script's entry point.
if __name__ == "__main__":
    main()