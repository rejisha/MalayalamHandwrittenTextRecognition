import argparse
import cv2
import os
import shutil
import difflib
from jiwer import cer, wer
from joblib import load

import pandas as pd
import numpy as np
import operator
from keras.models import load_model
from image_preprocessing import process_image, convert_to_transparent_bg
from segmentation  import process_image_segmentation


CHAR_DICT = {
    0: 'അ', 1: 'ആ', 2: 'ഇ', 3: 'ഉ', 4: 'എ', 5: 'ഏ', 6: 'ഒ',  
    7: 'ക', 8: 'ഖ', 9: 'ഗ', 10: 'ഘ', 11: 'ങ', 12: 'ച', 13: 'ഛ', 
    14: 'ജ', 15: 'ഝ', 16: 'ഞ', 17: 'ട', 18: 'ഠ', 19: 'ഡ', 20: 'ഢ', 
    21: 'ണ', 22: 'ത', 23: 'ഥ', 24: 'ദ', 25: 'ധ', 26: 'ന', 27: 'പ', 
    28: 'ഫ', 29: 'ബ', 30: 'ഭ', 31: 'മ', 32: 'യ', 33: 'ര', 34: 'റ', 
    35: 'ല', 36: 'ള', 37: 'ഴ', 38: 'വ', 39: 'ശ', 40: 'ഷ', 41: 'സ', 
    42: 'ഹ', 43: 'ൺ', 44: 'ൻ', 45: 'ർ', 46: 'ൽ', 47: 'ൾ', 48: 'ാ', 
    49: 'ി', 50: 'ീ', 51: 'ു', 52: 'ൂ', 53: 'െ', 54:' േ', 55: 'ൗ', 56:'മ്മ', 
    57: '്', 58: 'ല്ല', 59: 'ന്ന', 60: 'ട്ട', 61: 'ത്ത', 62: 'ണ്ട', 63: 'ക്ക', 64: 'ക്ഷ', 
}


SEGMENTED_FOLDER = 'segmented_characters/'
# MODEL = load_model("model_cnn_lstm.keras")
MODEL = load_model("model/cnn_model.keras")

# predcit character
def predict_character(image):

    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        img = convert_to_transparent_bg(img)

    img = process_image(img)
    img_data = img
    charset = np.asarray(img_data)
    charset = charset.reshape((-1, 32, 32, 1)).astype(np.float32)
    p = MODEL.predict(charset)[0]

    char_classes = np.genfromtxt('classes.csv', delimiter=',')[:, 1].astype(int)
    
    new = dict(zip(char_classes, p))
    res = sorted(new.items(), key=operator.itemgetter(1), reverse=True)
    character = int(res[0][0])

    if res[0][1] < 1:
        # print("Other predictions")
        for newtemp in res:
            character = newtemp[0]
            # print("Character = ", newtemp[0])
            # print("Confidence = ", newtemp[1] * 100, "%")

    return character

def image_to_text(input_img):
    
    result = process_image_segmentation(input_img, SEGMENTED_FOLDER)
    
    if result:
        char_files = os.listdir(SEGMENTED_FOLDER)
        # print(char_files)

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
                pred_character = predict_character(SEGMENTED_FOLDER + file)
                # print('pc ', predicted_character) 
                df = pd.read_csv('classes.csv', names=['Key', 'Value']) 
                class_num = df[df['Value'] == pred_character]['Key'].iloc[0]  
                print(' cls', class_num)  
                char = CHAR_DICT[class_num]
                if space:
                    text.append(' ')
                    text.append(char)  
                else: 
                    text.append(char)
            prev_filename = filename
        
            # os.remove(SEGMENTED_FOLDER+file)
        prev_filename = None

    predicted_text = ''.join(text)
    with open('predicted_text_svm.txt', 'w', encoding='utf-8') as f:
        f.write(predicted_text)
                
    return text


input_img = 'test_data/word_1.jpg'
# true_sentence =  'അമ്മ വയനശാലയിൽ പോയി '


text = image_to_text(input_img)
print(text)
# predicted_text = ''.join(text)

# # Calculate CER and WER
# print("CER:", cer(true_sentence, predicted_text))
# print("WER:", wer(true_sentence, predicted_text))




