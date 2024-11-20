**An Automated Approach to Detect Malayalam Handwriting Text**

**Overview**
This project focuses on recognizing handwritten Malayalam text using advanced models like CNN, CNN-LSTM, SVM, and GPT-4 Vision. It includes a web interface for digitizing handwritten Malayalam text into editable digital format.

**Features**
- OCR for Malayalam Script: Converts handwritten text to editable text.
- Model Comparisons: Evaluates SVM, CNN, CNN-LSTM, and GPT-4 Vision.
- Dataset Augmentation: Adds complex Malayalam characters.
- Demo Web App: Upload and process handwritten text images.

**How to Run:**
Install dependencies: pip install -r requirements.txt
Dataset Augmentation: python data_augumentation.py
Data cleaning: python data_cleaning.py
Processing images and generating a pickle file: python create_pickelfile.py
Train CNN model: python cnn_train.py
Train CNN-LTSM model: python cnnlstm_train.py
Train SVM model: python svm_model.py
Predict image: python predict.py
Predict image using svm: python predict_svm.py
Run GPT4Vision: python gpt4vision.py
Launch the web app: python app.py






