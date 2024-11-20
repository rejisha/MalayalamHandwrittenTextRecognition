import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, LSTM, Reshape, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report

# Load the data from the pickle file
PICKLE_FILE = 'data.pickle'

with open(PICKLE_FILE, 'rb') as pf:
    savefile = pickle.load(pf)
    traindataset = savefile['train_dataset']
    trainlabels = savefile['train_labels']
    testdataset = savefile['test_dataset']
    testlabels = savefile['test_labels']
    del savefile

BATCH_SIZE = 128
CLASSES = 65  
EPOCHS = 100

def lrate_schedule(ep):
    if ep < 50:
        return 0.001
    elif ep < 100:
        return 0.0005
    else:
        return 0.0001

lrate_scheduler = LearningRateScheduler(lrate_schedule)

def reformat_data(char_dataset, labels):
    char_dataset = char_dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    labels = (np.arange(CLASSES) == labels[:, None]).astype(np.float32)
    return char_dataset, labels

train_dataset, train_labels = reformat_data(traindataset, trainlabels)
test_dataset, test_labels = reformat_data(testdataset, testlabels)

# Model definition
cnnlstm_model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    Flatten(),
    Reshape(target_shape=(4, 256)),  # Assuming final feature map is 4x4x256
    LSTM(256, return_sequences=True),
    LSTM(256),
    Dropout(0.5),
    Dense(CLASSES, activation='softmax')
])

cnnlstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
cnnlstm_history = cnnlstm_model.fit(
    train_dataset, train_labels,
    steps_per_epoch=len(train_dataset) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(test_dataset, test_labels),
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True), lrate_scheduler]
)

# Evaluate the model
test_loss, test_accuracy = cnnlstm_model.evaluate(test_dataset, test_labels, verbose=1)

# Predict the labels for the test set
test_pred = cnnlstm_model.predict(test_dataset)
test_pred_top2 = np.argsort(test_pred, axis=1)[:, -2:]  # Get top 2 predictions
true_classes = np.argmax(test_labels, axis=1)

# Check if at least one of the top 2 predictions is correct
correct_top2 = [true_classes[i] in test_pred_top2[i] for i in range(len(true_classes))]
correct_top2_accuracy = np.mean(correct_top2)

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_classes, np.argmax(test_pred, axis=1))

# Print classification report
classification_report_str = classification_report(true_classes, np.argmax(test_pred, axis=1), digits=4)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cnnlstm_history.history['accuracy'], label='Training')
plt.plot(cnnlstm_history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.savefig('cnnlstm_model_accuracy.png')

plt.subplot(1, 2, 2)
plt.plot(cnnlstm_history.history['loss'], label='Training')
plt.plot(cnnlstm_history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('cnnlstm_model_loss.png')

plt.tight_layout()
plt.show()

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(CLASSES), yticklabels=np.arange(CLASSES))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

print("\nTraining Accuracy: {:.2f}%".format(cnnlstm_history.history['accuracy'][-1] * 100))
print("Validation (Test) Accuracy: {:.2f}%".format(cnnlstm_history.history['val_accuracy'][-1] * 100))
print('Test Accuracy:', test_accuracy)
print(f"Top-2 Accuracy: {correct_top2_accuracy:.4f}")
print(classification_report_str)

cnnlstm_model.save("new_model_cnn_lstm_3.keras")
