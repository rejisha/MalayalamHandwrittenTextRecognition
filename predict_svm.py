import cv2
from joblib import load



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


# Load the pre-fitted pipeline
model = load('svm_model.joblib')

def predict_character(img_path):
    # Load the image in grayscale
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image not found.")
        return None

    # Resize the image to the expected dimensions (32x32)
    image = cv2.resize(image, (32, 32))

    # Flatten the image to create a 1D array of features
    image_flattened = image.flatten().reshape(1, -1)  # Ensuring it's a 2D array with one sample

    prediction = model.predict(image_flattened)
    print(prediction)
    predicted_character = CHAR_DICT.get(int(prediction[0]), 'Unknown')
    
    return predicted_character

# Example usage
input_img = 'test_data/word_1.jpg'
predicted_char = predict_character(input_img)
print("Predicted Character:", predicted_char)
