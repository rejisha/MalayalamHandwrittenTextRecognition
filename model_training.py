import keras
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential


DATA_PATH = 'data.pickle'

# CNN Classifier
class CNNClassifier:
    def __init__(self, pickle_file, batch_size=60, classes=65, epochs=5000):
        self.pickle_file = pickle_file
        self.batch_size = batch_size
        self.classes = classes
        self.epochs = epochs
        self.model = None
        self.history = None

    def load_data(self):
        with open(self.pickle_file, 'rb') as pf:
            savefile = pickle.load(pf)
            self.train_dataset, self.train_labels = self.reformat_data(savefile['train_dataset'], savefile['train_labels'])
            self.valid_dataset, self.valid_labels = self.reformat_data(savefile['valid_dataset'], savefile['valid_labels'])
            self.test_dataset, self.test_labels = self.reformat_data(savefile['test_dataset'], savefile['test_labels'])
            del savefile

    def reformat_data(self, char_dataset, labels):
        char_dataset = char_dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
        labels = (np.arange(self.classes) == labels[:, None]).astype(np.float32)
        return char_dataset, labels

    def build_model(self):
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.classes, activation='softmax')
        ])
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

    def train(self):
        start_time = time.time()
        self.history = self.model.fit(self.train_dataset, self.train_labels,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      verbose=1,
                                      validation_data=(self.valid_dataset, self.valid_labels))
        end_time = time.time()
        print("[INFO] Total time taken to train the model: {:.2f}s".format(end_time - start_time))

    def evaluate(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_dataset, self.test_labels, verbose=1)
        train_loss, train_accuracy = self.model.evaluate(self.train_dataset, self.train_labels, verbose=0)
        val_loss, val_accuracy = self.model.evaluate(self.valid_dataset, self.valid_labels, verbose=0)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_accuracy)
        print('Training accuracy:', train_accuracy)
        print('Validation accuracy:', val_accuracy)

    def save_history(self, filename='cnn_training_history.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.history.history, f)

    def plot_metrics(self, filename='cnn_learning_curve.png'):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()  


cnn_classifier = CNNClassifier(DATA_PATH)
cnn_classifier.load_data()
cnn_classifier.build_model()
cnn_classifier.train()
cnn_classifier.evaluate()
cnn_classifier.save_history()
cnn_classifier.plot_metrics()

#SVM Classifier
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

class SVMClassifier:
    def __init__(self, data_path, kernel='rbf', C=1000, gamma=0.098):
        self.data_path = data_path
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, verbose=True)
        self.load_data()
        self.reshape_data()

    def load_data(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
            self.train_dataset = data['train_dataset']
            self.train_labels = data['train_labels']
            self.valid_dataset = data['valid_dataset']
            self.valid_labels = data['valid_labels']
            self.test_dataset = data['test_dataset']
            self.test_labels = data['test_labels']
            self.convert_labels()

    def reshape_data(self):
        self.train_dataset = self.train_dataset.reshape((self.train_dataset.shape[0], -1))
        self.valid_dataset = self.valid_dataset.reshape((self.valid_dataset.shape[0], -1))
        self.test_dataset = self.test_dataset.reshape((self.test_dataset.shape[0], -1))

    def convert_labels(self):
        if len(self.train_labels.shape) > 1 and self.train_labels.shape[1] > 1:
            self.train_labels = np.argmax(self.train_labels, axis=1)
            self.valid_labels = np.argmax(self.valid_labels, axis=1)
            self.test_labels = np.argmax(self.test_labels, axis=1)

    def fit(self):
        self.model.fit(self.train_dataset, self.train_labels)

    def predict_and_evaluate(self):
        train_predictions = self.model.predict(self.train_dataset)
        valid_predictions = self.model.predict(self.valid_dataset)
        test_predictions = self.model.predict(self.test_dataset)

        training_accuracy = accuracy_score(self.train_labels, train_predictions)
        validation_accuracy = accuracy_score(self.valid_labels, valid_predictions)
        test_accuracy = accuracy_score(self.test_labels, test_predictions)

        print('Training Accuracy:', training_accuracy)
        print('Validation Accuracy:', validation_accuracy)
        print('Test Accuracy:', test_accuracy)

        # Classification report for validation data
        print("Classification Report (Validation Data):")
        print(classification_report(self.valid_labels, valid_predictions))

    def save_model(self, filename='svm_model.joblib'):
        dump(self.model, filename)




classifier = SVMClassifier(data_path=DATA_PATH)
classifier.fit()
classifier.predict_and_evaluate()
classifier.save_model()


# CNN-LSTM Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, LSTM, Reshape, BatchNormalization
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CNNLSTMModel:
    def __init__(self, data_path, batch_size=128, classes=64, epochs=100):
        self.data_path = data_path
        self.batch_size = batch_size
        self.classes = classes
        self.epochs = epochs
        self.model = None
        self.history = None

        self.load_data()
        self.reformat_data()
        self.prepare_data_augmentation()
        self.compute_class_weights()
        self.build_model()

    def load_data(self):
        with open(self.data_path, 'rb') as pf:
            savefile = pickle.load(pf)
            self.train_dataset = savefile['train_dataset']
            self.train_labels = savefile['train_labels']
            self.valid_dataset = savefile['valid_dataset']
            self.valid_labels = savefile['valid_labels']
            self.test_dataset = savefile['test_dataset']
            self.test_labels = savefile['test_labels']
            del savefile

    def reformat_data(self):
        self.train_dataset, self.train_labels = self._reformat(self.train_dataset, self.train_labels)
        self.valid_dataset, self.valid_labels = self._reformat(self.valid_dataset, self.valid_labels)
        self.test_dataset, self.test_labels = self._reformat(self.test_dataset, self.test_labels)

    def _reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
        labels = (np.arange(self.classes) == labels[:, None]).astype(np.float32)
        return dataset, labels

    def prepare_data_augmentation(self):
        self.data_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest'
        )

    def compute_class_weights(self):
        true_class = np.argmax(self.train_labels, axis=1)
        cls_weights = compute_class_weight(
            'balanced',
            classes=np.unique(true_class),
            y=true_class.flatten()
        )
        self.cls_weights_dict = dict(enumerate(cls_weights))

    def build_model(self):
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Reshape(target_shape=(4, 128)),
            LSTM(256, return_sequences=True),
            LSTM(256),
            Dropout(0.5),
            Dense(self.classes, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self):
        lrate_scheduler = LearningRateScheduler(self.lrate_schedule)
        self.history = self.model.fit(
            self.data_gen.flow(self.train_dataset, self.train_labels, batch_size=self.batch_size),
            steps_per_epoch=len(self.train_dataset) // self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(self.valid_dataset, self.valid_labels),
            callbacks=[EarlyStopping(monitor='val_accuracy', patience=30), lrate_scheduler],
            class_weight=self.cls_weights_dict
        )

    def lrate_schedule(self, ep):
        if ep < 10:
            return 0.001
        elif ep < 20:
            return 0.0005
        else:
            return 0.0001

    def evaluate_model(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_dataset, self.test_labels, verbose=1)
        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_accuracy)

    def plot_results(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')
        plt.savefig('cnnlstm_model_accuracy.png')

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig('cnnlstm_model_loss.png')

        plt.tight_layout()
        plt.show()

    def save_model(self, filename="model_cnn_lstm.keras"):
        self.model.save(filename)


model = CNNLSTMModel(data_path=DATA_PATH)
model.train_model()
model.evaluate_model()
model.plot_results()
model.save_model()
