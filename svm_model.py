import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump


with open('data.pickle', 'rb') as f:
    data = pickle.load(f)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    test_labels = data['test_labels']


start_time = time.time()

# Reshape datasets for SVM (2D to 1D).
train_dataset = train_dataset.reshape((train_dataset.shape[0], -1))
test_dataset = test_dataset.reshape((test_dataset.shape[0], -1))

# One-hot encoded labels to single integers.
if len(train_labels.shape) > 1 and train_labels.shape[1] > 1:
    train_labels = np.argmax(train_labels, axis=1)
    test_labels = np.argmax(test_labels, axis=1)

scaler = StandardScaler()
train_dataset = scaler.fit_transform(train_dataset)
test_dataset = scaler.transform(test_dataset)


pca = PCA(n_components=0.85)  
train_dataset = pca.fit_transform(train_dataset)
test_dataset = pca.transform(test_dataset)

model = svm.SVC(kernel='rbf', C=100, gamma=0.098, verbose=True)
model.fit(train_dataset, train_labels)

# Predictions and accuracy calculations
train_pred = model.predict(train_dataset)
test_pred = model.predict(test_dataset)

training_accuracy = accuracy_score(train_labels, train_pred)
test_accuracy = accuracy_score(test_labels, test_pred)

end_time = time.time()
print("[INFO] total time taken to train the cnn_model: {:.2f}s".format(end_time - start_time))

print("Classification Report:")
print(classification_report(test_labels, test_pred))

print('Training Accuracy:', training_accuracy)
print('Test Accuracy:', test_accuracy)


dump(model, 'svm_model.joblib')


