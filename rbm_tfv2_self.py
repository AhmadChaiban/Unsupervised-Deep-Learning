import tensorflow as tf
from tensorflow import keras
import numpy as np

class NetworkTrainer:
    def __init__(self, train_dataset, test_dataset):
        self.train = train_dataset
        self.test = test_dataset

    def fit(self):

        model = self.network('rbm_forward', 'relu', 'relu', 'softmax')
        model.compile(optimizer='adam',
                      loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(
            self.train.repeat(),
            epochs=10,
            steps_per_epoch=500,
            validation_split=0.2,
            validation_steps=2
        )

    def network(self, activation1, activation2, activation3, out_activation):
        model = keras.Sequential([
            keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
            keras.layers.Dense(units=256, activation= activation1),
            keras.layers.Dense(units=192, activation= activation2),
            keras.layers.Dense(units=128, activation= activation3),
            keras.layers.Dense(units=10, activation= out_activation)
        ]);

        return model

def getMNIST():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return x_train, y_train, x_test, y_test

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)

    return x, y

def create_dataset(xs, ys, n_classes=10):
    ys = tf.one_hot(ys, depth=n_classes)
    return tf.data.Dataset.from_tensor_slices((xs, ys)) \
        .map(preprocess) \
        .shuffle(len(ys)) \
        .batch(128)

def RBM_forward(x):
    print('hello')
    print(x)

def RBM_both():
    pass

def RBM_back():
    pass

def main():

    keras.utils.generic_utils.get_custom_objects().update({'rbm_forward': keras.layers.Activation(RBM_forward)})

    X_train, y_train, X_test, y_test = getMNIST()
    train_dataset = create_dataset(X_train, y_train)
    test_dataset = create_dataset(X_test, y_test)

    network_trainer = NetworkTrainer(train_dataset, test_dataset)

    network_trainer.fit()

    # predictions = model.predict(val_dataset)
    # np.argmax(predictions[0])