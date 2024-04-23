import keras
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
# from six.moves import cPickle as Pickle
import pickle
import time

batch_size = 64
num_classes = 60
epochs = 10000

pickle_file = 'data.pickle'


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)
    return dataset, labels

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    # print('Training set', train_dataset.shape, train_labels.shape)
    # print('Validation set', valid_dataset.shape, valid_labels.shape)
    # print('Test set', test_dataset.shape, test_labels.shape)

start_time = time.time()

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

# print('Training set', train_dataset.shape, train_labels.shape)
# print('Validation set', valid_dataset.shape, valid_labels.shape)
# print('Testing set', test_dataset.shape, test_labels.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(train_dataset, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(valid_dataset, valid_labels))

test_loss, test_accuracy = model.evaluate(test_dataset, test_labels, verbose=1)

# Evaluate training dataset
training_loss, training_accuracy = model.evaluate(train_dataset, train_labels, verbose=0)

# Evaluate the model on the validation dataset
validation_loss, validation_accuracy = model.evaluate(valid_dataset, valid_labels, verbose=0)

model.save("model.keras")

print("[INFO] total time taken to train the model: {:.2f}s".format(time.time() - start_time))

print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
print('training_accuracy:', training_accuracy)
print('Validation Accuracy:', validation_accuracy)


