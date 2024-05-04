from image_preprocessing import *

def process_dataset(raw_folder, clean_folder):
    """
    Process a dataset by listing folders, creating necessary directories,
    and processing images from raw folders to clean folders.
    """
    folder_list = get_subdirectories(raw_folder)
    # print(folder_list)

    # Create corresponding folders in the clean data directory
    create_directory_structure(clean_folder, folder_list)

    # Process images from each raw folder to the corresponding clean folder
    process_and_save_images(raw_folder, clean_folder, folder_list)
    print('Process completed')


RAW_DATA_FOLDER = 'raw_data'
CLEANED_DATA_FOLDER = 'dataset'
process_dataset(RAW_DATA_FOLDER, CLEANED_DATA_FOLDER)
