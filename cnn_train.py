import keras
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential

import matplotlib.pyplot as plt
import pickle
import time

BATCH_SIZE = 128
CLASSES = 65
EPOCH = 5000

PICKLE_FILE = 'data.pickle'


def reformat_data(char_dataset, labels):
    '''
    Reshape the data into Keras input format 
    (num_samples, height, width, channels).
    '''
    char_dataset = char_dataset.reshape((-1, 32, 32, 1)).astype(np.float32) # 4D NumPy array.
    label = (np.arange(CLASSES)==labels[:, None]).astype(np.float32) # One-hot encoded label array, which matches each image to a specific class.
    return char_dataset, label

with open(PICKLE_FILE, 'rb') as pf:
    savefile = pickle.load(pf)
    traindataset = savefile['train_dataset']
    trainlabels = savefile['train_labels']
    testdataset = savefile['test_dataset']
    testlabels = savefile['test_labels']
    del savefile  

start_time = time.time()

# Data reshaping
train_dataset, train_labels = reformat_data(traindataset, trainlabels)
test_dataset, test_labels = reformat_data(testdataset, testlabels)

# CNN model
cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(CLASSES, activation='softmax'))

cnn_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

cnn_history = cnn_model.fit(train_dataset, train_labels,
          batch_size=BATCH_SIZE,
          epochs=EPOCH,
          verbose=1,
          validation_data=(test_dataset, test_labels))

with open('cnn_training_history.pkl', 'wb') as f:
    pickle.dump(cnn_history.history, f)

# Evaluate training, testing and validation 
test_loss, test_accuracy = cnn_model.evaluate(test_dataset, test_labels, verbose=1)
train_loss, train_accuracy = cnn_model.evaluate(train_dataset, train_labels, verbose=0)

cnn_model.save("cnn_model.keras")

end_time = time.time()
print("[INFO] total time taken to train the cnn_model: {:.2f}s".format(end_time - start_time))

print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
print('training_accuracy:', train_accuracy)

# Plot for accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'])
plt.plot(cnn_history.history['val_accuracy'])
plt.title('cnn_model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()

# Plot for loss
plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'])
plt.plot(cnn_history.history['val_loss'])
plt.title('cnn_model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('cnn_learning_curve.png')
plt.close() 
# plt.show()




