import os
import shutil
import Augmentor
from image_preprocessing import get_subdirectories

def clear_and_augment_images(data_folder):
    '''
    Image augmenation on the dataset
    '''
    for folder_name in get_subdirectories(data_folder):
        subfolder_path = os.path.join(data_folder, folder_name)
        
        if not os.path.isdir(subfolder_path):
            print(f"Skipping {subfolder_path}, not a directory.")
            continue

        output_dir_path = os.path.join(subfolder_path, 'output')
        
        if os.path.exists(output_dir_path):
            print(f"Removing existing output directory: {output_dir_path}")
            shutil.rmtree(output_dir_path)
        
        print(f"Starting augmentation in {subfolder_path}...")
        pipeline = Augmentor.Pipeline(subfolder_path)
        pipeline.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=8)
        pipeline.sample(500, multi_threaded=False)
        print(f"Augmentation completed for {subfolder_path}")


RAW_DATA_FOLDER = 'raw_data'
clear_and_augment_images(RAW_DATA_FOLDER)
