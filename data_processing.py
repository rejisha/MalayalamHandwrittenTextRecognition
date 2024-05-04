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
    class_folders = [os.path.join(path, d) for d in sorted(os.listdir(path))
                    if os.path.isdir(os.path.join(path, d))]

    if len(class_folders) != CLASS_COUNT:
        raise Exception(
            f'Expected {CLASS_COUNT} folders, one per class. Found {len(class_folders)} instead.')

    return class_folders


def load_class_images(folder, min_images):
    image_files = os.listdir(folder)
    images = np.ndarray(shape=(len(image_files), IMG_SIZE, IMG_SIZE), dtype=np.float32)
    loaded_images = 0
    for image in image_files:
        image_path = os.path.join(folder, image)
        try:
            img_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_data is None:
                raise IOError("File not readable")
            img_data = (img_data / PIXEL_SCALE).astype(np.float32)
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
    if num_rows:
        dataset = np.ndarray((num_rows, size, size), dtype=np.float32)
        labels = np.ndarray(num_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def combine_datasets(pickle_files, train_size, test_size=0, valid_size=0):
    class_total = len(pickle_files)
    valid_data, valid_labels = create_data_arrays(valid_size, IMG_SIZE)
    test_data, test_labels = create_data_arrays(test_size, IMG_SIZE)
    train_data, train_labels = create_data_arrays(train_size, IMG_SIZE)
    valid_per_class = valid_size // class_total
    test_per_class = test_size // class_total
    train_per_class = train_size // class_total

    start_v, start_t, start_tr = 0, valid_per_class, (valid_per_class + test_per_class)
    end_v, end_t, end_tr = valid_per_class, valid_per_class + test_per_class, train_per_class

    current_v, current_t, current_tr = 0, 0, 0
    next_v, next_t, next_tr = valid_per_class, test_per_class, train_per_class
    csv_rows = []
    for idx, file in enumerate(pickle_files):
        csv_rows.append([idx, file[-4:]])
        try:
            with open(file + PICKLE_EXT, 'rb') as f:
                image_set = pickle.load(f)
                np.random.shuffle(image_set)
                if valid_data is not None:
                    valid_data[current_v:next_v, :, :] = image_set[:end_v, :, :]
                    valid_labels[current_v:next_v] = idx
                    current_v += valid_per_class
                    next_v += valid_per_class

                if test_data is not None:
                    test_data[current_t:next_t, :, :] = image_set[start_t:end_t, :, :]
                    test_labels[current_t:next_t] = idx
                    current_t += test_per_class
                    next_t += test_per_class

                train_data[current_tr:next_tr, :, :] = image_set[start_tr:end_tr, :, :]
                train_labels[current_tr:next_tr] = idx
                current_tr += train_per_class
                next_tr += train_per_class
        except Exception as ex:
            print('Unable to process data from', file, ':', ex)
            raise
    with open('classes.csv', 'w') as my_csv:
        writer = csv.writer(my_csv, delimiter=',')
        writer.writerows(csv_rows)
    return valid_data, valid_labels, test_data, test_labels, train_data, train_labels


folders = list_class_folders(DATA_DIR)
prepared_datasets = prepare_pickle_files(folders, IMAGES_PER_CLASS, True)
train_count = int(IMAGES_PER_CLASS * CLASS_COUNT * 0.7)
test_count = int(IMAGES_PER_CLASS * CLASS_COUNT * 0.2)
valid_count = int(IMAGES_PER_CLASS * CLASS_COUNT * 0.1)

valid_dataset, valid_labels, test_dataset, test_labels, train_dataset, train_labels = combine_datasets(
    prepared_datasets, train_count, test_count, valid_count)

print('Training set:', train_dataset.shape, train_labels.shape)
print('Test set:', test_dataset.shape, test_labels.shape)
print('Validation set:', valid_dataset.shape, valid_labels.shape)

data_pickle = 'data.pickle'

try:
    with open(data_pickle, 'wb') as f:
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    print('Data saved to', data_pickle, 'Size:', os.stat(data_pickle).st_size)
except Exception as ex:
    print('Unable to save data:', ex)
    raise
