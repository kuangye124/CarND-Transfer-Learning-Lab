import pickle
import tensorflow as tf

# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Input
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    num_classes = len(np.unique(y_train))
    model = Sequential()

    # feature extraction using already trained model, thus
    # only needs to tune and add the last few layers
    model.add(Flatten(input_shape=(X_train[0].shape)))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # TODO: train your model here. y_train is not one-hot-coded
    # so use "sparse-categorical_crossentropy" instead
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

    history = model.fit(X_train, y_train,
                        batch_size=128,
                        epochs=10,
                        verbose=2,
                        validation_data=(X_val, y_val),
                        shuffle=True)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
