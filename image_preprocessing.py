import os
import cv2
import numpy as np

IMAGE_SIZE = 32

def get_subdirectories(root_path):
    """
    Get list of subdirectory names from the raw data foldeer.
    """
    subdirs = []
    for folder in sorted(os.listdir(root_path)):
        if os.path.isdir(os.path.join(root_path, folder)):
            subdirs.append(folder)
    return subdirs

def create_directory_structure(root_path, subdirs):
    """
    Creates directories based on a list of names within a root directory.
    """
    for subdir in subdirs:
        os.makedirs(os.path.join(root_path, subdir), exist_ok=True)

def convert_to_transparent_bg(filename):
    """
    Converts an image with an alpha channel to a BGR image with a white background.
    """
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image[:, :, 3] # extract tarnsparency
    rgb_channels = image[:, :, :3]
    white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
    alpha_factor = alpha_channel[..., np.newaxis].astype(np.float32) / 255.0 # normalize to range values between [1,0]
    foreground = rgb_channels.astype(np.float32) * alpha_factor
    background = white_background.astype(np.float32) * (1 - alpha_factor)
    final_image = foreground + background
    return final_image.astype(np.uint8)

def process_image(image):
    """
    Converts image to grayscale, applies binary threshold and extracts the largest contour.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]
    x, y, w, h = cv2.boundingRect(largest_contour)
    roi = binary_image[y:y+h, x:x+w]
    return transform_image(roi)

def transform_image(image):
    """
    Resizes, crops, and pads an image to a specified size.
    """
    old_size = image.shape[:2]  # (height, width)
    ratio = float(IMAGE_SIZE) / max(old_size)
    new_size = [int(x * ratio) for x in old_size]
    resized_image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = IMAGE_SIZE - new_size[1]
    delta_h = IMAGE_SIZE - new_size[0]
    padding = [(delta_h // 2, delta_h - delta_h // 2), (delta_w // 2, delta_w - delta_w // 2)]
    return cv2.copyMakeBorder(resized_image, *padding[0], *padding[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])

def process_images_in_folder(folder_path):
    """
    Processes all PNG images in a given folder.
    """
    images_processed = []
    for img_file in sorted(os.listdir(folder_path)):
        if img_file.lower().endswith('.png'):
            try:
                full_path = os.path.join(folder_path, img_file)
                image = convert_to_transparent_bg(full_path)
                processed_image = process_image(image)
                images_processed.append((img_file, processed_image))
            except Exception as e:
                print(f"Failed to process {img_file}: {e}")
    return images_processed

def save_processed_images(target_folder, images):
    """
    Saves processed images to a target directory.
    """
    for img_name, img_data in images:
        cv2.imwrite(os.path.join(target_folder, img_name), img_data)

def process_and_save_images(source_folder, target_folder, subdirectories):
    """
    Process images from subdirectories and save them to a corresponding subfolders.
    """
    for subdir in subdirectories:
        print("Processing: ", subdir)
        images = process_images_in_folder(os.path.join(source_folder, subdir, 'output'))
        save_processed_images(os.path.join(target_folder, subdir), images)

def skeletonize(image):
    """
    Converts a binary image to its skeleton representation.
    """
    skeleton = np.zeros(image.shape, np.uint8)
    struct_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(image, struct_element)
        temp = cv2.dilate(eroded, struct_element)
        temp = cv2.subtract(image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()
        if cv2.countNonZero(image) == 0:
            done = True
    return skeleton