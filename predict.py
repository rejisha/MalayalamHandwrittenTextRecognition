import argparse
import cv2
import os
import shutil
import difflib

import pandas as pd
import numpy as np
import operator
from keras.models import load_model
from functions import clean, read_transparent_png
from segmentation import image_segmentation


CHAR_DICT = {
    0: 'അ', 1: 'ആ', 2: 'ഇ', 3: 'ഉ', 4: 'എ', 5: 'ഏ', 6: 'ഒ',  
    7: 'ക', 8: 'ഖ', 9: 'ഗ', 10: 'ഘ', 11: 'ങ', 12: 'ച', 13: 'ഛ', 
    14: 'ജ', 15: 'ഝ', 16: 'ഞ', 17: 'ട', 18: 'ഠ', 19: 'ഡ', 20: 'ഢ', 
    21: 'ണ', 22: 'ത', 23: 'ഥ', 24: 'ദ', 25: 'ധ', 26: 'ന', 27: 'പ', 
    28: 'ഫ', 29: 'ബ', 30: 'ഭ', 31: 'മ', 32: 'യ', 33: 'ര', 34: 'റ', 
    35: 'ല', 36: 'ള', 37: 'ഴ', 38: 'വ', 39: 'ശ', 40: 'ഷ', 41: 'സ', 
    42: 'ഹ', 43: 'ൺ', 44: 'ൻ', 45: 'ർ', 46: 'ൽ', 47: 'ൾ', 48: 'ാ', 
    49: 'ി', 50: 'ീ', 51: 'ു', 52: 'ൂ', 53: 'െ', 54: 'േ', 55: 'ൗ', 56:'മ്മ', 
    57: '്', 58: 'ല്ല', 59: '൦', 
}


SEGMENTED_FOLDER = 'segmented_characters/'

# Function to predict characters from a given image
def predict_character(img):

    print('img ', img)
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    model = load_model("model.keras")
    # model = load_model("model_cnn_lstm.keras")

    if image.shape[2] == 4:
        image = read_transparent_png(image)

    image = clean(image)
    # cv2.imshow('gray', image)
    # cv2.waitKey(0)

    # Predict characters from the cleaned image
    image_data = image
    dataset = np.asarray(image_data)
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    a = model.predict(dataset)[0]
    # print(a)

    classes = np.genfromtxt('classes.csv', delimiter=',')[:, 1].astype(int)
    # print(classes)

    new = dict(zip(classes, a))
    res = sorted(new.items(), key=operator.itemgetter(1), reverse=True)

    print("#########***#########")
    character = int(res[0][0])
    # print("Character = ", character)
    # print("Confidence = ", res[0][1] * 100, "%")

    if res[0][1] < 1:
        # print("Other predictions")
        for newtemp in res:
            character = newtemp[0]
            # print("Character = ", newtemp[0])
            # print("Confidence = ", newtemp[1] * 100, "%")

    return character

# Parse command line arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
# args = vars(ap.parse_args())


def image_to_text(input_img):
    if os.path.splitext(input_img)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
        # Perform word segmentation and save the segmented character images
        result = image_segmentation(input_img, SEGMENTED_FOLDER)
        
        if result:
            char_files = os.listdir(SEGMENTED_FOLDER)
            # print(char_files)
            # Predict characters for each segmented character image
            text = []
            prev_filename = None
            for file in char_files:  
                space = False
                filename = os.path.splitext(file)[0]
                if prev_filename:
                    split_prevfilename = prev_filename.split('_')
                    split_filename = filename.split('_')
                    if split_filename[1] != split_prevfilename[1]:
                        space = True
                        print('there is a space between characters')
                if file is not None:
                    predicted_character = predict_character(SEGMENTED_FOLDER + file)
                    print('pc ', predicted_character) 
                    df = pd.read_csv('classes.csv', names=['Key', 'Value']) 
                    class_num = df[df['Value'] == predicted_character]['Key'].iloc[0]  
                    print(' cls', class_num)  
                    char = CHAR_DICT[class_num]
                    if space:
                        text.append(' ')
                        text.append(char)  
                    else: 
                        text.append(char)
                prev_filename = filename
            
                os.remove(SEGMENTED_FOLDER+file)
            prev_filename = None

    predicted_text = ''.join(text)
    with open('predicted_text.txt', 'w', encoding='utf-8') as f:
        f.write(predicted_text)
                
    return predicted_text


def edit_distance(str1, str2):
    s = difflib.SequenceMatcher(None, str1, str2)
    return s.ratio()

def calculate_cer(reference, pred_text):
    reference = reference.replace(" ", "")  # Remove spaces for character level comparison
    pred_text = pred_text.replace(" ", "")
    return (1 - edit_distance(reference, pred_text)) * 100

def calculate_wer(reference, pred_text):
    ref_words = reference.split()
    hyp_words = pred_text.split()

    substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
    deletions = len(ref_words) - len(hyp_words)
    insertions = len(hyp_words) - len(ref_words)
    # Total number of words in the reference text
    total_words = len(ref_words)
    # Calculating the Word Error Rate (WER)
    wer = (substitutions + deletions + insertions) / total_words
    return wer

input_img = 'test_data/test_sen_2.jpg'
true_sentence =  'ഇന്ന് പണിമുടക്'


# predicted_text = image_to_text(input_img)
# predicted_text = ''.join(text)
# print(predicted_text)
# # Calculate CER and WER
# print("CER:", calculate_cer(true_sentence, predicted_text))
# print("WER:", calculate_wer(true_sentence, predicted_text))


# with open('predicted_text.txt', 'w', encoding='utf-8') as f:
#     # Write the joined string to the file
#     f.write(predicted_text)
# print(text)
