import keras
import numpy as np
import pickle
import time
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, LSTM, Reshape
from keras.models import Sequential

# Load data
pickle_file = 'data.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # Free up memory

# Constants
batch_size = 64
num_classes = 60
epochs = 10000


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)
    return dataset, labels

# Reformat data
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Prepare output for LSTM layers
model.add(Reshape(target_shape=(4, 128)))  # Adjusted for output of last MaxPooling2D

# Add LSTM layers
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))

# Classification layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Train the model
start_time = time.time()
history = model.fit(train_dataset, train_labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_dataset, valid_labels))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset, test_labels, verbose=1)
training_loss, training_accuracy = model.evaluate(train_dataset, train_labels, verbose=0)
validation_loss, validation_accuracy = model.evaluate(valid_dataset, valid_labels, verbose=0)

# Save the model
model.save("model_cnn_lstm.keras")

print("[INFO] total time taken to train the model: {:.2f}s".format(time.time() - start_time))
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
print('Training accuracy:', training_accuracy)
print('Validation accuracy:', validation_accuracy)
