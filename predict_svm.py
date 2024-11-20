import os
import cv2
import numpy as np
from joblib import load
from segmentation import process_image_segmentation

SEGMENTED_FOLDER = 'segmented_characters/'
svm_model = load('new_svm_model.joblib')

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

def predict_character_svm(input_img):
    '''
    Preprocess the input image for SVM prediction.
    '''
    img = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))  # Ensure the image is 32x32
    img = img / 255.0  # Normalize the image
    img = img.flatten().astype(np.float32)  # Flatten the image

    prediction = svm_model.predict([img])
    return prediction[0]


# Function to convert segmented images into text
def image_to_text(input_img):
 
    result = process_image_segmentation(input_img, SEGMENTED_FOLDER)
    
    if result:
        char_files = sorted(os.listdir(SEGMENTED_FOLDER))
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
                    print('There is a space between characters')
            
            pred_character = predict_character_svm(SEGMENTED_FOLDER + file)
            char = CHAR_DICT.get(pred_character, '')
            if space:
                text.append(' ')
            text.append(char)
            
            prev_filename = filename
            os.remove(SEGMENTED_FOLDER + file)
        
        prev_filename = None

    predicted_text = ''.join(text)
    with open('predicted_text_svm.txt', 'w', encoding='utf-8') as f:
        f.write(predicted_text)
                
    return predicted_text



input_img = 'test_data\\sen_5.jpg'
predicted_character = image_to_text(input_img)

