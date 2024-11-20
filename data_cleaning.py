from image_preprocessing import *


def process_dataset(source_folder, clean_folder):
    """
    processing images from raw folders to clean folder
    """
    folder_list = get_subdirectories(source_folder)
    # print(folder_list)

    # Create corresponding folders in the dataset folder
    create_directory_structure(clean_folder, folder_list)

    # Process images from each raw folder to the corresponding dataset subfolder
    process_and_save_images(source_folder, clean_folder, folder_list)
    print('Process completed')


SOURCE_DATA_FOLDER = 'source_data'
CLEANED_DATA_FOLDER = 'dataset'
process_dataset(SOURCE_DATA_FOLDER, CLEANED_DATA_FOLDER)
