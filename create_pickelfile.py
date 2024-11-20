import os
import cv2
import numpy as np
import pickle
import csv

DATA_DIR = 'dataset'
IMG_SIZE = 32
PIXEL_SCALE = 255
PICKLE_EXT = '.pickle'
CLASS_COUNT = 65
IMAGES_PER_CLASS = 500

def list_class_folders(path):
    '''
    List folders from dataset folder.
    '''
    class_folders = [os.path.join(path, d) for d in sorted(os.listdir(path))
                    if os.path.isdir(os.path.join(path, d))]

    if len(class_folders) != CLASS_COUNT:
        raise Exception(
            f'Expected {CLASS_COUNT} folders, one per class. Found {len(class_folders)} instead.')

    return class_folders

def load_class_images(folder, min_images):
    '''
    Load and process images from dataset folder.
    '''
    image_files = os.listdir(folder)
    images = np.ndarray(shape=(len(image_files), IMG_SIZE, IMG_SIZE), dtype=np.float32)
    loaded_images = 0
    for image in image_files:
        image_path = os.path.join(folder, image)
        try:
            img_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_data is None:
                raise IOError("File not readable")
            img_data = (img_data / PIXEL_SCALE).astype(np.float32) # Divide the image pixel by PIXEL_SCALE to normalize them 
            if img_data.shape != (IMG_SIZE, IMG_SIZE):
                raise Exception(f'Unexpected image shape: {img_data.shape}')
            images[loaded_images, :, :] = img_data
            loaded_images += 1
        except Exception as ex:
            print(f'Could not process {image_path}: {ex} - skipping.')

    if loaded_images < min_images:
        raise Exception(f'Many fewer images than expected: {loaded_images} < {min_images}')
    
    images = images[:loaded_images, :, :]
    return images

def prepare_pickle_files(class_folders, min_images_per_class, overwrite=False):
    '''
    Preparing pickle file for each class.
    '''
    pickle_files = []
    for folder in class_folders:
        pickle_filename = folder + PICKLE_EXT
        pickle_files.append(folder)
        if os.path.exists(pickle_filename) and not overwrite:
            print(f'{pickle_filename} already present - Skipping pickling.')
        else:
            class_images = load_class_images(folder, min_images_per_class)
            try:
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(class_images, f, pickle.HIGHEST_PROTOCOL)
            except Exception as ex:
                print('Unable to save data to', pickle_filename, ':', ex)

    return pickle_files

def create_data_arrays(num_rows, size):
    '''
    Create arrays to store the dataset and its corresponding labels.
    '''
    if num_rows:
        dataset = np.ndarray((num_rows, size, size), dtype=np.float32)
        labels = np.ndarray(num_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def combine_datasets(pickle_files, train_size=0, test_size=0):
    '''
    Create pickle files for training and testing datasets, and
    create a CSV for mapping class indices to labels.
    '''

    class_total = len(pickle_files)

    # Arrays to store the dataset and its corresponding labels.
    test_data, test_labels = create_data_arrays(test_size, IMG_SIZE)
    train_data, train_labels = create_data_arrays(train_size, IMG_SIZE)

    # Number of images allocated per class for testing and training.
    test_per_class = test_size // class_total
    train_per_class = train_size // class_total

    csv_rows = []
    for idx, file in enumerate(pickle_files):
        csv_rows.append([idx, file[-4:]]) # String to extract the last four characters,
        try:
            with open(file + PICKLE_EXT, 'rb') as f:
                image_set = pickle.load(f)
                np.random.shuffle(image_set)

                num_images = len(image_set)
                train_end = int(num_images * 0.8)
                test_end = num_images

                if train_data is not None:
                    train_data[idx * train_per_class: (idx + 1) * train_per_class, :, :] = image_set[:train_per_class, :, :]
                    train_labels[idx * train_per_class: (idx + 1) * train_per_class] = idx

                if test_data is not None:
                    test_data[idx * test_per_class: (idx + 1) * test_per_class, :, :] = image_set[train_per_class:test_per_class + train_per_class, :, :]
                    test_labels[idx * test_per_class: (idx + 1) * test_per_class] = idx

        except Exception as ex:
            print('Unable to process data from', file, ':', ex)
            raise

    with open('classes.csv', 'w') as my_csv:
        writer = csv.writer(my_csv, delimiter=',')
        writer.writerows(csv_rows)
    return test_data, test_labels, train_data, train_labels

folders = list_class_folders(DATA_DIR) #list the character folders from the dataset.
prepared_datasets = prepare_pickle_files(folders, IMAGES_PER_CLASS, True)
train_count = int(IMAGES_PER_CLASS * CLASS_COUNT * 0.8) # 1000*60*0.8 = 48000
test_count = int(IMAGES_PER_CLASS * CLASS_COUNT * 0.2) # 1000*60*0.2 = 12000

test_dataset, test_labels, train_dataset, train_labels = combine_datasets(
    prepared_datasets, train_count, test_count)

print('Training set:', train_dataset.shape, train_labels.shape)
print('Test set:', test_dataset.shape, test_labels.shape)

data_pickle = 'data.pickle'

try:
    with open(data_pickle, 'wb') as f:
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    print('Data saved to', data_pickle, 'Size:', os.stat(data_pickle).st_size)
except Exception as ex:
    print('Unable to save data:', ex)
    raise
