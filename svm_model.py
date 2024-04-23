import pickle
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score

# Load the data
pickle_file = 'data.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # free up memory

# Flatten the datasets for SVM (since SVM expects 2D array [n_samples, n_features])
train_dataset = train_dataset.reshape((train_dataset.shape[0], -1))
valid_dataset = valid_dataset.reshape((valid_dataset.shape[0], -1))
test_dataset = test_dataset.reshape((test_dataset.shape[0], -1))

# Check if labels are one-hot encoded and convert if necessary
if len(train_labels.shape) > 1 and train_labels.shape[1] > 1:
    train_labels = np.argmax(train_labels, axis=1)
    valid_labels = np.argmax(valid_labels, axis=1)
    test_labels = np.argmax(test_labels, axis=1)

# Standardize the data
scaler = StandardScaler()
train_dataset = scaler.fit_transform(train_dataset)
valid_dataset = scaler.transform(valid_dataset)
test_dataset = scaler.transform(test_dataset)

# Create and train the SVM model
model = svm.SVC(kernel='linear', C=1.0, gamma=0.02, verbose=True)
model.fit(train_dataset, train_labels)


# # Evaluate the model on training data
train_predictions = model.predict(train_dataset)
training_accuracy = accuracy_score(train_labels, train_predictions)
print('Training Accuracy:', training_accuracy)

# # Evaluate the model
valid_predictions = model.predict(valid_dataset)
validation_accuracy = accuracy_score(valid_labels, valid_predictions)
print('Validation Accuracy:', validation_accuracy)

test_predictions = model.predict(test_dataset)
test_accuracy = accuracy_score(test_labels, test_predictions)
print('Test Accuracy:', test_accuracy)


# # Optionally save the model
from joblib import dump
dump(model, 'svm_model.joblib')
