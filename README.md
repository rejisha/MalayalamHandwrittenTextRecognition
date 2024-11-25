# **An Automated Approach to Detect Malayalam Handwriting Text**

## **Overview**
This project focuses on recognizing handwritten Malayalam text using advanced models like CNN, CNN-LSTM, SVM, and GPT-4 Vision. It includes a web interface for digitizing handwritten Malayalam text into editable digital format.

---

## Dataset

### Download the Dataset
The dataset required for this project is hosted on Google Drive. You can download it using the following link:

[Download Dataset](https://drive.google.com/drive/folders/1vcruciA1Ef9PggMM8Ix07OuBLc-oVg-P?usp=sharing)

---

## **Features**
- **OCR for Malayalam Script**: Converts handwritten text to editable text.
- **Model Comparisons**: Evaluates SVM, CNN, CNN-LSTM, and GPT-4 Vision.
- **Dataset Augmentation**: Adds complex Malayalam characters.
- **Demo Web App**: Upload and process handwritten text images.

---

## **How to Run**

- Install dependencies:  
   ```bash
   pip install -r requirements.txt

- Augment the dataset: `python data_augumentation.py`

- Clean the data: `python data_cleaning.py`

- Process images and generate a pickle file: `python create_pickelfile.py`

- Train the CNN model: `python cnn_train.py`

- Train the CNN-LSTM model: `python cnnlstm_train.py`

- Train the SVM model: `python svm_model.py`

- Predict an image: `python predict.py`

- Predict an image using the SVM model: `python predict_svm.py`

- Run GPT-4 Vision: `python gpt4vision.py`

- Launch the web application: `python app.py`




